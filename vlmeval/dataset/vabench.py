import re
import ast
import json
import math
import os.path as osp
import pandas as pd
from PIL import Image
from ..smp import * # 假设环境中有此库
from .image_base import ImageBaseDataset

# 复用之前的 smart_resize 逻辑
# 注意：如果你的代码在一个文件中，可以直接调用；如果在不同文件，请确保导入
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

class VABench(ImageBaseDataset):
    TYPE = "VQA"

    def __init__(
        self,
        dataset="VABench",
        data_file="VABench",
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
        # 过滤掉没有图片的样本
        if skip_noimg and "image_paths" in data.columns:
            data = data[data["image_paths"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]

        self.meta_only = True 

        if "id" in data.columns:
            data["index"] = data["id"]
        
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
        
        # --- 针对 Qwen2.5-VL 的 Prompt 增强 ---
        # 你的数据已经是 Robot 指令，但如果需要强制 JSON 格式，保持这个逻辑是安全的
        if "qwen2.5" in self.model_type:
            instruction = "Please detect the specific points and return their coordinates in JSON format. " \
                          "Example: ```json\n[{\"point_2d\": [x, y]}, ...]\n```"
        else:
            instruction = "Please detect the specific points and return their normalized [0-1000] coordinates in JSON format. " \
                          "Example: ```json\n[{\"point_2d\": [x, y]}, ...]\n```"
        question = question + " " + instruction

        visual_indices = line["visual_indices"]
        pil_images = tgt_path
        
        # 构建图文交错的 content 列表
        contents = []
        if len(visual_indices) == 0:
            for img in pil_images:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
        else:
            # 简单的处理逻辑：图片放在前面，文本在最后（根据你的数据 visual_indices=[0]）
            # 如果需要更复杂的插值逻辑可以复用 RefSpatialBench 的代码
            for img in pil_images:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
        
        return contents

    def get_image_size(self, image_path):
        with Image.open(image_path) as img:
            return img.height, img.width

    def normalize_prediction(self, x, y, image_path):
        """
        统一将预测坐标转换为 0-1 相对坐标
        """
        if "qwen2.5" in self.model_type:
            if not image_path or not osp.exists(image_path):
                return x, y 
            
            orig_h, orig_w = self.get_image_size(image_path)
            resize_h, resize_w = smart_resize(orig_h, orig_w)
            
            norm_x = x / resize_w if resize_w > 0 else 0
            norm_y = y / resize_h if resize_h > 0 else 0
            
            return min(max(norm_x, 0.0), 1.0), min(max(norm_y, 0.0), 1.0)
        else:
            if x > 1.0:
                return x / 1000.0, y / 1000.0
            return x, y

    @staticmethod
    def extract_points_from_json(text):
        """
        专门解析 [{"point_2d": [x, y]}, ...] 格式
        返回: [[x1, y1], [x2, y2], ...]
        """
        text = str(text).strip()
        points = []
        
        # 1. 尝试提取 Markdown JSON 块
        match = re.search(r"```json\n(.*)\n```", text, re.DOTALL)
        json_str = match.group(1).strip() if match else text
        
        try:
            # 修复末尾逗号等
            json_str_fixed = re.sub(r',\s*([\]\}])', r'\1', json_str)
            data = json.loads(json_str_fixed)
            
            if isinstance(data, dict): data = [data]
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "point_2d" in item:
                        pt = item["point_2d"]
                        if isinstance(pt, list) and len(pt) >= 2:
                            points.append([float(pt[0]), float(pt[1])])
        except:
            # Fallback: 简单的正则提取 [num, num]
            matches = re.findall(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', json_str)
            for m in matches:
                points.append([float(m[0]), float(m[1])])
        
        return points

    @staticmethod
    def extract_points_from_tuples(text):
        """
        专门解析 [(x1, y1), (x2, y2), ...] 格式
        返回: [[x1, y1], [x2, y2], ...]
        """
        if not text:
            return []
            
        text = str(text).strip()
        points = []
        
        # 1. 预处理：提取 Markdown 代码块内的内容（如果存在）
        match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text)
        content = match.group(1).strip() if match else text
        
        # 2. 尝试使用 ast.literal_eval (最安全且能解析原生 Python 列表/元组)
        try:
            # 清理掉可能存在的 "Output:" 等前缀字符串，寻找第一个 '[' 或 '('
            start_idx = re.search(r'[\[\(]', content)
            if start_idx:
                potential_list = content[start_idx.start():]
                data = ast.literal_eval(potential_list)
                
                # 如果解析出来是单个元组，转为列表处理
                if isinstance(data, tuple) and len(data) == 2 and not isinstance(data[0], (list, tuple)):
                    data = [data]
                    
                if isinstance(data, (list, tuple)):
                    for item in data:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            points.append([float(item[0]), float(item[1])])
                    if points: return points
        except (ValueError, SyntaxError, TypeError):
            pass

        # 3. Fallback: 强大的正则提取
        # 匹配模式: (数字, 数字)
        # 支持: 整数、浮点数、负数、科学计数法
        num_pattern = r'(-?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        tuple_pattern = rf'\(\s*{num_pattern}\s*,\s*{num_pattern}\s*\)'
        
        matches = re.findall(tuple_pattern, content)
        for m in matches:
            try:
                points.append([float(m[0]), float(m[1])])
            except ValueError:
                continue
                
        return points

    def check_point_in_bbox(self, norm_point, bbox, image_path):
        """
        检查 0-1 归一化坐标是否在原始 BBox 内
        """
        if norm_point is None or bbox is None:
            return False
            
        # 获取原图尺寸用于还原绝对坐标
        try:
            orig_h, orig_w = self.get_image_size(image_path)
        except Exception:
            return False

        # 将归一化坐标还原为原图绝对像素坐标
        norm_x, norm_y = norm_point
        x_px = norm_x * 1000
        y_px = norm_y * 1000
        
        # 解析 bbox [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = bbox
        
        # 判断是否在范围内
        return (x1 <= x_px <= x2) and (y1 <= y_px <= y2)

    def evaluate(self, eval_file, **judge_kwargs):
        data_pred = load(eval_file)
        if "index" not in data_pred.columns and "id" in data_pred.columns:
             data_pred["index"] = data_pred["id"]
        
        pred_map = dict(zip(data_pred["index"], data_pred["prediction"]))

        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}

        for _, row in self.data.iterrows():
            idx = row["index"]
            if idx not in pred_map:
                continue

            prediction = str(pred_map[idx])
            
            # 获取 BBox
            bbox = row.get("bbox", None) # Expected list: [x1, y1, x2, y2]
            
            # 获取图片路径
            image_path = None
            if "image_paths" in row:
                rel_paths = row["image_paths"]
                if isinstance(rel_paths, list) and len(rel_paths) > 0:
                     image_path = osp.join(self.data_root, rel_paths[0])
            
            if image_path is None or bbox is None:
                continue

            task = row.get("question_type", "unknown")
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1

            # 1. 提取点
            raw_points = self.extract_points_from_json(prediction)
            if not raw_points:
                raw_points = self.extract_points_from_tuples(prediction)
            
            # 2. 验证: 只要有一个点命中 BBox 即算正确 (Hit Rate)
            is_correct = False
            if raw_points:
                for (rx, ry) in raw_points:
                    # 先归一化（处理 Qwen2.5 的 Smart Resize）
                    norm_point = self.normalize_prediction(rx, ry, image_path)
                    # 再反算回原图与 BBox 比较
                    if self.check_point_in_bbox(norm_point, bbox, image_path):
                        is_correct = True
                        break

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