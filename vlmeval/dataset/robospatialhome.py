import re
import ast
import json
import math
import os.path as osp
from PIL import Image
import pandas as pd
import numpy as np

from ..smp import *
from .image_base import ImageBaseDataset

# --- Qwen2.5-VL Smart Resize Logic (内置) ---
IMAGE_FACTOR = 28
MIN_PIXELS = 64 * 28 * 28
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

class RoboSpatialHomeBench(ImageBaseDataset):
    TYPE = "VQA"

    def __init__(
        self,
        dataset="RoboSpatialHomeBench",
        data_file="RoboSpatialHomeBench",
        data_root=None,
        skip_noimg=True,
        model_type="qwen2.5-vl",
    ):
        self.dataset_name = dataset
        self.data_file = data_file
        self.data_root = data_root
        self.model_type = model_type.lower()

        data = self.load_data(data_file)

        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        # 必须改为 True，因为我们需要在 evaluate 时读取原图尺寸
        self.meta_only = True 

        if "id" in data.columns:
            data["index"] = data["id"]
        
        # 建立索引映射
        self.index_map = {str(row['index']): row for _, row in data.iterrows()}
        
        self.data = data
        self.post_build(self.dataset_name)

    def load_data(self, data_file):
        def load_jsonl(f):
            lines = open(f, encoding="utf-8").readlines()
            lines = [x.strip() for x in lines]
            if lines and lines[-1] == "":
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
        
        if "yes or no" in question:
            pass
        else:
            # --- 适配 Qwen2.5-VL 的 Prompt 格式 ---
            if "qwen2.5" in self.model_type:
                # 查找原始 prompt 中的格式说明部分进行替换，或者直接追加
                # 这里采用了追加/强化的策略，确保模型看到 point_2d 格式要求
                instruction = "Please detect the specific points and return their coordinates in JSON format. " \
                            "Example: ```json\n[{\"point_2d\": [x1, y1]}, {\"point_2d\": [x2, y2]}]\n```"
                
                if "json" not in question.lower():
                    split_marker = "Your answer should be formatted"
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
        return contents

    def get_image_size(self, image_path):
        with Image.open(image_path) as img:
            return img.height, img.width

    def normalize_prediction(self, x, y, image_path):
        """核心逻辑：坐标空间转换"""
        # 1. 普通模式 (0-1000 归一化)
        if "qwen2.5" not in self.model_type:
            return x / 1000.0, y / 1000.0
        
        # 2. Qwen2.5 Smart Resize 绝对坐标模式
        if not image_path or not osp.exists(image_path):
            return 0, 0 # Error fallback

        orig_h, orig_w = self.get_image_size(image_path)
        resize_h, resize_w = smart_resize(orig_h, orig_w)
        
        # 绝对坐标 -> 0-1 相对坐标
        norm_x = x / resize_w if resize_w > 0 else 0
        norm_y = y / resize_h if resize_h > 0 else 0
        
        return min(max(norm_x, 0.0), 1.0), min(max(norm_y, 0.0), 1.0)

    @staticmethod
    def extract_points_from_json(text):
        """
        专门解析 Qwen2.5-VL 的 [{"point_2d": [x, y]}, ...] 格式
        返回 list of [x, y]
        """
        text = str(text)
        points = []
        
        # 1. 尝试提取 Markdown JSON 块
        match = re.search(r"```json\n(.*)\n```", text, re.DOTALL)
        json_str = match.group(1).strip() if match else text
        
        try:
            # 修复常见 JSON 错误 (如 trailing commas)
            json_str_fixed = re.sub(r',\s*([\]\}])', r'\1', json_str)
            data = json.loads(json_str_fixed)
            
            if isinstance(data, dict):
                data = [data]
                
            if isinstance(data, list):
                for item in data:
                    # 提取 point_2d
                    if isinstance(item, dict) and "point_2d" in item:
                        pt = item["point_2d"]
                        if isinstance(pt, list) and len(pt) >= 2:
                            points.append([float(pt[0]), float(pt[1])])
                    # 提取 bbox_2d (如果混入，取中心点或忽略，这里暂忽略)
                    
        except json.JSONDecodeError:
            # Fallback: 正则暴力提取 [x, y]
            # 注意：这可能会匹配到 bbox 的前两个数，需谨慎
            matches = re.findall(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', json_str)
            for m in matches:
                points.append([float(m[0]), float(m[1])])
                
        return points

    @staticmethod
    def point_in_polygon(x, y, poly):
        num = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, num + 1):
            p2x, p2y = poly[i % num]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def evaluate_answer(self, ground_truth, generated_answer, image_path=None):
        gen_answer = str(generated_answer).strip().lower()
        gt_lower = str(ground_truth).strip().lower()
        
        # 1. Binary Yes/No Check
        if gt_lower in ["yes", "no"]:
            is_gt_yes = (gt_lower == "yes")
            if is_gt_yes:
                return gen_answer.startswith("yes") or gen_answer.startswith("y")
            else:
                return gen_answer.startswith("no") or gen_answer.startswith("n")

        # 2. Spatial Point Evaluation
        try:
            # 解析 GT Polygon (RoboSpatial 标准: 0-1000 整数)
            gt_polygon_raw = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon_raw, list) or len(gt_polygon_raw) < 3:
                return False
            
            # GT 转 0-1
            gt_polygon_norm = [(p[0]/1000.0, p[1]/1000.0) for p in gt_polygon_raw]

            # 解析预测点列表
            pred_points = self.extract_points_from_json(generated_answer)
            
            # 如果 JSON 解析失败，尝试旧的单个元组正则兜底
            if not pred_points:
                match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', gen_answer)
                if match:
                    pred_points.append([float(match.group(1)), float(match.group(2))])
                else:
                    match_list = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', gen_answer)
                    if match_list:
                        pred_points.append([float(match_list.group(1)), float(match_list.group(2))])

            if not pred_points:
                return False

            # 3. 验证：只要有一个预测点在 GT 多边形内，即视为正确 (Any Hit)
            if "qwen2.5" in self.model_type and (image_path is None or not osp.exists(image_path)):
                print(f"Warning: Image path missing for resize calc: {image_path}")
                return False

            for (px, py) in pred_points:
                # 归一化
                nx, ny = self.normalize_prediction(px, py, image_path)
                # 判定
                if self.point_in_polygon(nx, ny, gt_polygon_norm):
                    return True

            return False

        except Exception as e:
            # print(f"Error evaluating answer: {e}")
            return False
            
        return False

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data["prediction"] = [str(x) for x in data["prediction"]]
        
        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}

        for i, row in data.iterrows():
            task = row.get("question_type", "unknown")
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1
            
            ground_truth = row["answer"]
            prediction = row["prediction"]
            
            # 获取图片路径用于计算 resize
            idx_key = str(row.get("index", row.get("id", "")))
            image_path = None
            if idx_key in self.index_map:
                rel_paths = self.index_map[idx_key]["image_paths"]
                if isinstance(rel_paths, list) and len(rel_paths) > 0:
                    image_path = osp.join(self.data_root, rel_paths[0])
            
            is_correct = self.evaluate_answer(ground_truth, prediction, image_path=image_path)
            
            if is_correct:
                task_stats[task]["correct"] += 1
                micro_metric["correct"] += 1

        accuracy_dict = {
            task: [stats["correct"] / stats["total"] if stats["total"] > 0 else 0] 
            for task, stats in sorted(task_stats.items())
        }
        
        result_df = pd.DataFrame(accuracy_dict)
        if not result_df.empty:
            result_df["Overall macro"] = result_df.mean(axis=1)
            result_df["Overall micro"] = micro_metric["correct"] / micro_metric["total"] if micro_metric["total"] > 0 else 0
        
        suffix = eval_file.split(".")[-1]
        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(result_df, score_file)
        
        return result_df