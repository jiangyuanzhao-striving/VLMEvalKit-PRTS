import re
import ast
import json
import math
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional

from ..smp import *
from .image_base import ImageBaseDataset

# --- Qwen2.5-VL Smart Resize Logic (内置) ---
IMAGE_FACTOR = 28
MIN_PIXELS = 64 * 28 * 28  # 注意: 这里的最小/最大像素值按需调整，保持跟官方一致
MAX_PIXELS = 128 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        pass 
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

class RoboRefitBench(ImageBaseDataset):
    TYPE = "GROUNDING"
    MODALITY = 'IMAGE'
    IOU_THRESHOLD = 0.5

    def __init__(
        self,
        dataset="RoboRefitBench",
        data_file="RoboRefitBench",
        data_root=None,
        skip_noimg=True,
        model_type="qwen2.5-vl",  # 新增: 控制坐标解析模式
    ):
        self.dataset_name = dataset
        self.data_file = data_file
        self.data_root = data_root
        self.model_type = model_type.lower()

        data = self.load_data(data_file)

        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        # 必须为 True，因为 smart_resize 需要读取原图尺寸
        self.meta_only = True

        data["index"] = data["id"]
        
        # 建立索引映射，方便 evaluate 时获取图片路径
        self.index_map = {str(row['index']): row for _, row in data.iterrows()}

        self.data = data
        self.post_build(self.dataset_name)

    def load_data(self, data_file="RoboRefitBench"):
        def load_jsonl(f):
            lines = open(f, encoding="utf-8").readlines()
            lines = [x.strip() for x in lines]
            if lines[-1] == "":
                lines = lines[:-1]
            data = [json.loads(x) for x in lines]
            return pd.DataFrame(data)

        if not data_file.endswith(".jsonl") and osp.exists(data_file + ".jsonl"):
            data_file = data_file + ".jsonl"
            
        data = load_jsonl(data_file)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_paths"])
            tgt_path = [osp.join(self.data_root, p) for p in tgt_path]
        else:
            tgt_path = self.dump_image(line)

        question = line["question"]
        
        # --- Qwen2.5-VL Prompt 增强 ---
        if "qwen2.5" in self.model_type:
            instruction = " Please output the bounding boxes as a list of absolute pixel coordinates in JSON format."
            split_marker = "Your answer should be"
            if split_marker in question:
                # 尝试替换原有的格式说明
                base_q = question.split(split_marker)[0].strip()
                question = base_q + " " + instruction
            else:
                # 直接追加
                question = question + " " + instruction

        visual_indices = line["visual_indices"]
        pil_images = tgt_path
        image_index_pairs = list(zip(pil_images, visual_indices))
        image_index_pairs.sort(key=lambda x: x[1])

        contents = []
        if len(visual_indices) == 0:
            for img in pil_images:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
        elif all(idx == 0 for idx in visual_indices):
            for img, _ in image_index_pairs:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
        else:
            last_pos = 0
            for img, idx in image_index_pairs:
                if idx == 0:
                    contents.append(dict(type="image", value=img))
                else:
                    if idx <= len(question):
                        text_segment = question[last_pos:idx]
                        if text_segment:
                            contents.append(dict(type="text", value=text_segment))
                        contents.append(dict(type="image", value=img))
                        last_pos = idx
                    else:
                        contents.append(dict(type="image", value=img))
            if last_pos < len(question):
                contents.append(dict(type="text", value=question[last_pos:]))
            if not contents:
                contents.append(dict(type="text", value=question))
                for img, _ in image_index_pairs:
                    contents.append(dict(type="image", value=img))

        return contents
    
    @staticmethod
    def extract_bbox_from_json_string(text):
        """
        增强版 JSON 提取，提取 bbox_2d 字段
        """
        text = str(text).strip()
        
        # 1. 尝试 Markdown Block
        match = re.search(r"```json\n(.*)\n```", text, re.DOTALL)
        json_candidates = []
        if match:
            json_candidates.append(match.group(1).strip())
        
        # 2. 尝试寻找 raw JSON 结构
        # 匹配 [{"bbox_2d": ...}]
        match_raw = re.search(r'(\[?\s*\{.*?"bbox_2d".*?\}\s*\]?)', text, re.DOTALL)
        if match_raw:
             json_candidates.append(match_raw.group(1).strip())
        
        for json_str in json_candidates:
            try:
                # 修复逗号
                json_str_fixed = re.sub(r',\s*([\]\}])', r'\1', json_str)
                data = json.loads(json_str_fixed)
                
                if isinstance(data, dict): data = [data]
                
                if isinstance(data, list) and data:
                    item = data[0]
                    if isinstance(item, dict) and 'bbox_2d' in item:
                        return item['bbox_2d']
            except:
                continue
        
        # Fallback: 如果没有 JSON，不强行解析，返回 None
        return None
    
    @staticmethod
    def _standard_to_absolute(coords: List, width: float, height: float) -> Optional[np.ndarray]:
        """处理 0-1 或 0-1000 归一化坐标"""
        if coords is None or len(coords) < 4 or width <= 0 or height <= 0:
            return None

        coords = np.array(coords).astype(float)[:4]
        if np.any(np.isnan(coords)): return None

        max_val = float(np.max(np.abs(coords)))
        if max_val <= 1.5:
            coords[0::2] *= width
            coords[1::2] *= height
        elif max_val <= 1000:
            coords[0::2] = coords[0::2] / 1000.0 * width
            coords[1::2] = coords[1::2] / 1000.0 * height
        
        return RoboRefitBench._clip_and_sort(coords, width, height)

    @staticmethod
    def _qwen25_to_absolute(coords: List, width: float, height: float) -> Optional[np.ndarray]:
        """
        Qwen2.5 专用: 处理 Smart Resize 后的绝对坐标
        """
        if coords is None or len(coords) < 4 or width <= 0 or height <= 0:
            return None
        
        coords = np.array(coords).astype(float)[:4]
        if np.any(np.isnan(coords)): return None
        
        # 计算 Resize 后的画布尺寸
        resize_h, resize_w = smart_resize(height, width)
        
        # 映射回原图绝对坐标
        if resize_w > 0:
            coords[0::2] = coords[0::2] / resize_w * width
        if resize_h > 0:
            coords[1::2] = coords[1::2] / resize_h * height
            
        return RoboRefitBench._clip_and_sort(coords, width, height)

    @staticmethod
    def _clip_and_sort(coords, width, height):
        coords[0] = np.clip(coords[0], 0, width)
        coords[2] = np.clip(coords[2], 0, width)
        coords[1] = np.clip(coords[1], 0, height)
        coords[3] = np.clip(coords[3], 0, height)

        if coords[2] < coords[0]:
            coords[0], coords[2] = coords[2], coords[0]
        if coords[3] < coords[1]:
            coords[1], coords[3] = coords[3], coords[1]
        return coords

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data["prediction"] = [str(x) for x in data["prediction"]]
        
        # 确保 index 类型一致
        if "index" not in data.columns and "id" in data.columns:
            data["index"] = data["id"]
        data['index'] = data['index'].astype(str)

        pred_bboxes: List[str] = []
        ious: List[float] = []
        hits: List[int] = []

        for index, row in data.iterrows():
            idx_key = row["index"]
            
            # 1. 获取原图尺寸 (优先从 result file 获取，没有则查 index_map)
            width = row.get("image_width", 0)
            height = row.get("image_height", 0)
            
            if (width <= 0 or height <= 0) and idx_key in self.index_map:
                meta_row = self.index_map[idx_key]
                # 尝试从 meta 获取
                if "image_width" in meta_row and not pd.isna(meta_row["image_width"]):
                    width = meta_row["image_width"]
                    height = meta_row["image_height"]
                # 尝试从图片读取 (最后的手段)
                elif "image_paths" in meta_row:
                    try:
                        img_path = osp.join(self.data_root, meta_row["image_paths"][0])
                        with Image.open(img_path) as img:
                            width, height = img.size
                    except:
                        pass
            
            # 2. 解析预测框
            raw_pred = self.extract_bbox_from_json_string(row["prediction"])
            if "qwen2.5" in self.model_type:
                pred = self._qwen25_to_absolute(raw_pred, width=width, height=height)
            else:
                pred = self._standard_to_absolute(raw_pred, width=width, height=height)

            # 3. 解析 GT (假设 GT 始终是数据集标准的 0-1000 或 0-1，不随模型变)
            raw_gt = self.extract_bbox_from_json_string(row["answer"])
            answer = self._standard_to_absolute(raw_gt, width=width, height=height)

            # 4. 计算指标
            if pred is not None and answer is not None:
                iou = float(self._compute_iou(np.array(pred), np.array(answer)))
            else:
                iou = 0.0
                
            hit = 1 if iou >= self.IOU_THRESHOLD else 0
            hits.append(hit)
            ious.append(iou)
            pred_bboxes.append(self._format_bbox(pred))

        data['pred_bbox'] = pred_bboxes
        data['iou'] = ious
        data['hit'] = hits

        summary_rows: List[Dict[str, object]] = []
        overall_row = {
            'Split': 'Average',
            'Precision@1': float(np.mean(hits)) * 100 if hits else 0.0,
            'Average IoU': float(np.mean(ious)) if ious else 0.0,
            'Samples': len(hits),
        }
        summary_rows.append(overall_row)
        summary_df = pd.DataFrame(summary_rows)
        
        suffix = eval_file.split(".")[-1]
        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(summary_df, score_file)
        
        return summary_df

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        inter = (x_right - x_left) * (y_bottom - y_top)
        area1 = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
        area2 = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union
    
    @staticmethod
    def _format_bbox(bbox: Optional[np.ndarray]) -> str:
        if bbox is None:
            return ''
        return '[{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*bbox)