from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from transformers import StoppingCriteria

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY
from .constants import (
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    IMAGE_PLACEHOLDER_TOKEN,
    STATE_START_TOKEN,
    STATE_END_TOKEN,
    STATE_PLACEHOLDER_TOKEN,
    FAST_TRIGGER_TOKEN,
    EXPERT_TRIGGER_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    ACTION_PLACEHOLDER_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    SYSTEM_MESSAGE,
    IGNORE_INDEX,
)


VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

UNTIL = ["<|diff_marker|>"]


class PRTSVLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = SYSTEM_MESSAGE

        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path

        from .config import TrainConfig
        from .configuration_prts import PRTS_FlowMatchingConfig
        from .modeling_prts import PRTS
        from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from .processing_prts import PRTS_Processor
        from .processing_action_tokenizer import UniversalActionProcessor

        # training_args = TrainConfig()
        # config = PRTS_FlowMatchingConfig.from_pretrained(
        #     model_path
        # )
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_path, 
        #     config=config,
        #     torch_dtype='auto', device_map="auto", attn_implementation='flash_attention_2'
        # )        
        # self.processor = PRTS_Processor.from_pretrained(
        #     model_path,
        #     padding_side="right",
        #     use_fast=True
        # )
        # self.update_processor_pixels(training_args)


        # load prts
        ## load processor
        self.training_args = TrainConfig()
        self.processor = PRTS_Processor.from_pretrained(
            model_path,
            padding_side="right",
            use_fast=True,
        )
        self.action_tokenizer = UniversalActionProcessor.from_pretrained(
            f"{os.environ['HF_HUB_CACHE']}/models--physical-intelligence--fast", 
            trust_remote_code=True
        )
        self.processor.set_action_tokenizer(self.action_tokenizer)

        ## load config
        self.config = PRTS_FlowMatchingConfig.from_pretrained(
            model_path,
        )

        self.model = PRTS.from_pretrained(
            model_path,
            config=self.config,
            dtype="auto",
            device_map="auto",
        )       # model.config._attn_implementation 都是 sdpa （默认值）


        self.model.set_fast_action_info(self.processor.action_mapper, self.processor.action_token_start_index)
        self.update_processor_pixels(self.training_args)

        self.model.eval()

        torch.cuda.empty_cache()


    def update_processor_pixels(self, data_args):
        # --- Image Processor ---
        ip = self.processor.image_processor
        print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
        print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
        print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
        print(f"ip.size: {ip.size}")
        print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
        print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

        if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
            ip.min_pixels = data_args.image_min_pixels
            ip.max_pixels = data_args.image_max_pixels
            print(f"✅ Updated image_processor min_pixels to {data_args.image_min_pixels}")
            print(f"✅ Updated image_processor max_pixels to {data_args.image_max_pixels}")

        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = data_args.image_min_pixels
            ip.size["longest_edge"] = data_args.image_max_pixels
            print(
                f"✅ Updated image_processor size['shortest_edge'] to {data_args.image_min_pixels}"
            )
            print(
                f"✅ Updated image_processor size['longest_edge'] to {data_args.image_max_pixels}"
            )

        print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
        print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
        print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
        print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
        print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

        # --- Video Processor ---
        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
            vp = self.processor.video_processor
            print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
            print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
            print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
            print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
            print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
            print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
            print(
                f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

            if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
                vp.min_pixels = data_args.video_min_pixels
                vp.max_pixels = data_args.video_max_pixels
                print(
                    f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
                )
                print(
                    f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
                )

            if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
                vp.min_frames = data_args.video_min_frames
                vp.max_frames = data_args.video_max_frames
                print(
                    f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
                )
                print(
                    f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
                )

            if hasattr(vp, "fps"):
                vp.fps = data_args.fps
                print(f"✅ Updated video_processor fps to {data_args.fps}")

            if hasattr(vp, "size") and isinstance(vp.size, dict):
                vp.size["shortest_edge"] = data_args.video_min_pixels
                vp.size["longest_edge"] = data_args.video_max_pixels
                print(
                    f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
                )
                print(
                    f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
                )

            print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
            print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
            print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
            print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
            print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
            print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
            print(
                f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        # print(self.processor.tokenizer.batch_decode(
        #     inputs["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
        # ))
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response


    def generate_inner(self, message, dataset=None):

        return self.generate_inner_transformers(message, dataset=dataset)
