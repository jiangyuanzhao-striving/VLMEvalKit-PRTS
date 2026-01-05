# Copyright 2025 TeleAI Rhodes Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main VLA model architecture based on Qwen2.5-VL."""


from dataclasses import dataclass

import os
import math
import glob
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from safetensors.torch import load_file
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import Qwen2_5_VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import restore_default_dtype, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling

from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from .configuration_prts import PRTS_VLTextConfig, PRTS_FlowMatchingConfig
# from .subgoal_generator import SubGoalImageGenerator, SubGoalGenerationConfig, create_subgoal_generator

'''
Sinusoidal positional embedding for diffusion timesteps.
'''
def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device="cpu",
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

@dataclass
class PRTSModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    flow_loss: Optional[torch.FloatTensor] = None
    cross_entropy_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

    channel_loss_dict: Optional[dict[torch.FloatTensor]] = None
    channel_loss_count_dict: Optional[dict[torch.FloatTensor]] = None


## VLA built by TeleAI Rhodes Team - Primitive Reasoning and Tasking System (PRTS)
class PRTS(Qwen2_5_VLForConditionalGeneration):
    """
    Vision-Language-Action model based on Qwen2.5-VL.
    
    This model extends Qwen2.5-VL to support:
    1. Sub-task description generation (language format)
    2. Sub-goal prediction for subtasks (image format)
    3. Action chunk prediction (pi0 and pi0.5 styles)
    """
    config: PRTS_FlowMatchingConfig

    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]


    def __init__(
        self,
        config: PRTS_FlowMatchingConfig,
        use_fast_tokenizer: bool = True,
        flow_matching_action_loss_weight: float = 1.0,
        flow_matching_sub_goal_loss_weight: float = 1.0,
    ):
        """
        Initialize the Qwen2.5 VLMoE model for action processing.

        Args:
            config: Model configuration
            use_fast_tokenizer (bool): Whether to use FAST tokenizer
            processor: Text and image processor
            flow_matching_action_loss_weight (float): Weight for flow matching action loss computation
            flow_matching_sub_goal_loss_weight (float): Weight for flow matching subgoal loss computation
        """
        super().__init__(config)

        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel._from_config(config.text_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # for state projection
        self.action_dim = config.max_action_dim
        self.state_proj = nn.Linear(self.action_dim * 2, config.text_config.hidden_size, bias=False)

        self.loss_fct = CrossEntropyLoss(reduction="none")
        
        # mimic pi0
        ## Initialize the continuous action prediction modules and action experts
        #### TODO (zy): use GR00T like DiT in the future
        if flow_matching_action_loss_weight > 0.:
            # self.fm_action_module = FlowMatchingActionModule(config)
            self.fm_action_expert = Qwen2_5_VLModel._from_config(config.action_expert_config)
            # self.fm_action_expert.lm_head = None
            self.fm_action_expert.embed_tokens = None

        # Action embedding network: project to hidden space
        self.action_in_proj = nn.Linear(
            self.action_dim * 2, config.action_expert_config.hidden_size, bias=False
        )  # *2 for action + DOF mask
        self.action_time_mlp_in = nn.Linear(
            config.action_expert_config.hidden_size * 2, config.action_expert_config.hidden_size, bias=False
        )  # *2 for action + time embeddings
        self.action_time_mlp_out = nn.Linear(config.action_expert_config.hidden_size, config.action_expert_config.hidden_size, bias=False)
        
        self.action_out_proj = nn.Linear(
            config.action_expert_config.hidden_size, self.action_dim, bias=False
        )

        self.loss_mse = nn.MSELoss(reduction="none")

        ## Initialize the coefficient of every loss function
        self.flow_matching_action_loss_weight = flow_matching_action_loss_weight
        self.flow_matching_sub_goal_loss_weight = flow_matching_sub_goal_loss_weight
        self.use_fast_tokenizer = use_fast_tokenizer

        # TODO (zy): incorporate the subgoal in & out projection
        # self.subgoal_modules = xxx

        # Initialize weights and apply final processing
        self.post_init()

        # Print parameter counts
        model_params = sum(p.numel() for p in self.model.parameters())
        vision_params = sum(p.numel() for p in self.visual.parameters())
        print(f"Backbone VLM (self.model) parameters: {(model_params + vision_params) / 1e6:.2f}M")
        print(f"Flow Matching Loss coef: {self.flow_matching_action_loss_weight}")

        if flow_matching_action_loss_weight > 0.:
            expert_params = sum(p.numel() for p in self.fm_action_expert.parameters())
            print(f"Action Expert (self.fm_action_expert) parameters: {expert_params / 1e6:.2f}M")

        self.fast_action_token_start_idx = 200000

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def to_float32_flow_matching_head(self):
        self.action_out_proj = self.action_out_proj.to(dtype=torch.float32)
        self.action_time_mlp_in = self.action_time_mlp_in.to(dtype=torch.float32)
        self.action_time_mlp_out = self.action_time_mlp_out.to(dtype=torch.float32)
        self.state_proj = self.state_proj.to(dtype=torch.float32)
        self.action_in_proj = self.action_in_proj.to(dtype=torch.float32)

    def set_fast_action_info(self, action_mapper, fast_action_token_start_idx):
        self.action_mapper = action_mapper
        self.fast_action_token_start_idx = fast_action_token_start_idx
    
    def sample_noise(self, shape, device, dtype):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise.to(dtype)
    
    def sample_time(self, bsize, device, dtype):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=dtype)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_continuous_action(self, time, noisy_action_chunk, action_dof_mask=None):
        device = noisy_action_chunk.device

        noisy_action_chunk = noisy_action_chunk

        if action_dof_mask is not None:
            action_dof_mask = action_dof_mask.to(noisy_action_chunk.dtype)
            noisy_action = torch.cat([noisy_action_chunk, action_dof_mask], dim=-1)

        # Project action + dof mask to hidden size
        action_embeds = self.action_in_proj(noisy_action)

        # Project time to hidden size
        time_embeds = create_sinusoidal_pos_embedding(
            time,
            self.config.action_expert_config.hidden_size,
            device=device,
        )
        time_embeds = time_embeds.to(dtype=self.action_time_mlp_in.weight.dtype)
        time_embeds = time_embeds[:, None, :].expand_as(action_embeds)

        # Concatenate action and time embeddings
        action_time_embs = torch.cat([action_embeds, time_embeds], dim=2)
        action_time_embs = self.action_time_mlp_in(action_time_embs)
        action_time_embs = F.silu(action_time_embs)
        action_time_embs = self.action_time_mlp_out(action_time_embs)

        return action_time_embs

    # deprecated
    def get_vision_language_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract vision-language features from Qwen2.5-VL backbone."""
        
        # Get the hidden states from Qwen2.5-VL
        outputs = self.qwen_model.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use the last hidden state
        hidden_states = outputs.last_hidden_state
        
        return hidden_states
    
    
    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        state_dof_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # token -> embeddings, if inputs_embeds = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # processing image
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # processing video
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # processing proprioceptive state
        if states is not None:
            states = states.to(inputs_embeds.device).to(dtype=self.state_proj.weight.dtype)
            state_dof_mask = state_dof_mask.to(inputs_embeds.device).to(dtype=self.state_proj.weight.dtype)
            state_proj_input = torch.cat([states, state_dof_mask], dim=-1)
            state_embeds = self.state_proj(state_proj_input)
            state_mask = self.get_placeholder_mask_with_special_token(
                input_ids, inputs_embeds=inputs_embeds,
                special_features=state_embeds,
                special_pad_token_id=self.config.state_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_embeds)

        return inputs_embeds
    
    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        state_dof_mask: Optional[torch.Tensor] = None,
        action_dof_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # prepare position_ids and kv_cache
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

        # embed prefix
        if inputs_embeds is None:
            inputs_embeds = self.embed_prefix(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                states=states,
                state_dof_mask=state_dof_mask,
            )

        # pass prefix, update kvcache
        # TODO: 从这里开始需要修改，定制化内容
        chunk_size = self.config.action_chunk_size
        # suffix_len = -1  # exclude <|action_end|>
        # prefix_len = seq_len - chunk_size - 1

        # here we suppose that the length of attention is equal to the one of input_ids
        # NOTE: the attention mask does not contain how we deal with the actions
        cache_seq_len = attention_mask.shape[-1]

        outputs = self.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=cache_position if cache_position is not None else None,
        )

        # denoising
        device = states.device
        actions_shape = (inputs_embeds.shape[0], chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device=device, dtype=states.dtype)

        x_t = noise.to(self.action_in_proj.weight.dtype)
        dt = torch.tensor(-1.0 / self.config.num_denoise_steps, device=device)
        time = torch.ones(inputs_embeds.shape[0], device=device, dtype=states.dtype)
        past_key_values, past_hidden_state = outputs.past_key_values, outputs.last_hidden_state

        # action_mask = input_ids == self.config.action_token_id

        # generate position_ids and attention_mask
        context_len = torch.max(position_ids, dim=-1, keepdim=True)[0]
        action_position_ids = torch.arange(
            chunk_size, device=inputs_embeds.device
        ).view(1, 1, -1).repeat(3, inputs_embeds.shape[0], 1)
        action_position_ids += context_len

        expert_kwargs = {}
        if "flash_attention" in self.config._attn_implementation:
            action_attention_mask = torch.cat(
                [attention_mask, torch.ones((inputs_embeds.shape[0], chunk_size), device=inputs_embeds.device, dtype=torch.long)], dim=-1
            )
            expert_kwargs["is_causal"] = False
        else:
            prefix_att_mask = attention_mask[:, None, None, :].expand(-1, 1, chunk_size, -1)
            action_attention_mask = torch.cat(
                [prefix_att_mask, torch.ones((inputs_embeds.shape[0], 1, chunk_size, chunk_size), device=inputs_embeds.device, dtype=torch.bool)], dim=-1
            )

        expert_cache_position = None
        if cache_position is not None and context_len is not None:
            context_len_for_cache = context_len[0].squeeze(-1) # Shape: (selected_batch_size)
            expert_cache_position = cache_position + context_len_for_cache


        while time >= -dt / 2:
            action_time_embs = self.embed_continuous_action(time, x_t, action_dof_mask)
            # inputs_embeds[action_mask] = action_time_embs.to(inputs_embeds.dtype)
            expert_outputs = self.fm_action_expert(
                position_ids=action_position_ids,
                attention_mask=action_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=action_time_embs,
                use_cache=False,
                cache_position=expert_cache_position if expert_cache_position is not None else None,
            )

            past_key_values.crop(cache_seq_len)  # 只取前cache_prefix_len个

            action_hidden_states = expert_outputs.last_hidden_state
            action_hidden_states = action_hidden_states.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(action_hidden_states)

            x_t += dt * v_t.reshape(x_t.shape)
            time += dt
        
        outputs.last_hidden_state = torch.cat([past_hidden_state, outputs.last_hidden_state], dim=1)
        return x_t, outputs

    
    # def prepare_inputs_for_generation(self, *args, **kwargs):
    #     return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # def _expand_inputs_for_generation(self, *args, **kwargs):
    #     return self.model._expand_inputs_for_generation(*args, **kwargs)


PRTS.register_for_auto_class()
