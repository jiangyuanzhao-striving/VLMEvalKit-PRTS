import re
import json
import os.path as osp
import pandas as pd
from collections import defaultdict

from ..smp import *
from .image_base import ImageBaseDataset


class EmbSpatialBench(ImageBaseDataset):
    TYPE = "MCQ"

    def __init__(
        self,
        dataset="EmbSpatialBench",
        data_file="EmbSpatialBench",
        data_root=None,
        skip_noimg=True,
    ):
        # 覆盖默认数据集名称
        self.dataset_name = dataset
        self.data_file = data_file
        self.data_root = data_root

        data = self.load_data(data_file)

        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        self.meta_only = True

        # 确保 index 列存在，用于后续 evaluate 时的对齐
        if "id" in data.columns:
            data["index"] = data["id"]
        elif "index" not in data.columns:
             data["index"] = range(len(data))

        self.data = data
        self.post_build(self.dataset_name)

    def load_data(self, data_file):
        """
        加载数据，支持自动补全 .jsonl 后缀
        """
        def load_jsonl(f):
            lines = open(f, encoding="utf-8").readlines()
            lines = [x.strip() for x in lines]
            if lines and lines[-1] == "":
                lines = lines[:-1]
            data = [json.loads(x) for x in lines]
            return pd.DataFrame(data)
        
        # 兼容性处理：如果文件名没有后缀且存在对应 .jsonl 文件，自动补全
        if not data_file.endswith(".jsonl") and osp.exists(data_file + ".jsonl"):
            data_file = data_file + ".jsonl"
            
        data = load_jsonl(data_file)
        return data

    def build_prompt(self, line):
        """
        构建多模态 Prompt，支持图文交错 (Interleaved Image-Text)
        逻辑完全复用 ERQABench，因为它非常健壮
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_paths"])
            tgt_path = [osp.join(self.data_root, p) for p in tgt_path]
        else:
            tgt_path = self.dump_image(line)

        question = line["question"]
        visual_indices = line["visual_indices"]

        # 准备图片
        pil_images = tgt_path

        # 创建 (image, index) 对并排序
        image_index_pairs = list(zip(pil_images, visual_indices))
        image_index_pairs.sort(key=lambda x: x[1])

        contents = []

        # Case 1: 没有 visual_indices (图片在前，文本在后)
        if len(visual_indices) == 0:
            for img in pil_images:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
        
        # Case 2: 所有索引都是 0 (图片在前，文本在后)
        elif all(idx == 0 for idx in visual_indices):
            for img, _ in image_index_pairs:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
            
        # Case 3: 正常的图文穿插
        else:
            last_pos = 0
            for img, idx in image_index_pairs:
                # if idx == 0:
                contents.append(dict(type="image", value=img))
                # else:
                #     # 添加该图片之前的文本片段
                #     if idx <= len(question):
                #         text_segment = question[last_pos:idx]
                #         if text_segment:
                #             contents.append(dict(type="text", value=text_segment))
                #         contents.append(dict(type="image", value=img))
                #         last_pos = idx
                #     else:
                #         contents.append(dict(type="image", value=img))

            # 添加剩余文本
            if last_pos < len(question):
                contents.append(dict(type="text", value=question[last_pos:]))

            # 兜底：如果上面逻辑没添加任何内容，采用默认图前文后
            if not contents:
                contents.append(dict(type="text", value=question))
                for img, _ in image_index_pairs:
                    contents.append(dict(type="image", value=img))

        return contents

    @staticmethod
    def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]):
        """
        从模型输出中提取选项字母 (A, B, C, D...)
        """
        if isinstance(s, dict):
            s = ""
        s = str(s).strip()
        
        # 移除常见的回答前缀，防止干扰提取
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option is",
            "The correct option is",
            "Best answer:",
            "Best option:",
            "Answer:",
        ]
        for answer_prefix in answer_prefixes:
            # 忽略大小写替换
            if s.lower().startswith(answer_prefix.lower()):
                s = s[len(answer_prefix):].strip()

        # 尝试提取
        # 1. 匹配 (A) 格式
        matches = re.findall(r"\(([a-gA-G])\)", s)
        
        # 2. 匹配 A. 或 A 结尾 格式
        if len(matches) == 0:
            matches = re.findall(r"(?:^|\s)([a-gA-G])(?:$|[\s,.])", s)
        
        # 3. 匹配任意 A-G 字符 (最宽松，但可能会误判单词里的字母)
        # 通常前两步够用，这一步可视情况保留或删除
        if len(matches) == 0:
            # 限制只匹配开头，防止匹配到句子中间的单词
            matches = re.findall(r"^([a-gA-G])", s)

        if len(matches) == 0:
            return ""
        
        # 取第一个匹配到的作为答案 (单选)
        return matches[0].upper()

    def evaluate(self, eval_file, **judge_kwargs):
        """
        评估函数：计算准确率
        """
        data = load(eval_file)
        data["prediction"] = [str(x) for x in data["prediction"]]
        
        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}
        
        for index, it in data.iterrows():
            task = it.get("question_type", "Unknown") # 兼容缺失 question_type
            
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1
            
            # 提取预测答案和GT
            pred = self.extract_characters_regex(it["prediction"])
            gt = str(it["answer"]).strip().upper()
            
            # 判定正确性
            if pred == gt:
                task_stats[task]["correct"] += 1
                micro_metric["correct"] += 1
        
        # 计算各 Task 准确率
        accuracy_dict = {
            task: [stats["correct"] / stats["total"]] 
            for task, stats in sorted(task_stats.items())
        }
        
        result_df = pd.DataFrame(accuracy_dict)
        
        # 计算 Sphere/Category 宏平均 (如果 task 名字包含 '/')
        # 例如: "Spatial/Relation" -> Sphere = "Spatial"
        sphere_accs = defaultdict(list)
        for task, acc in accuracy_dict.items():
            sphere = task.split("/")[0]
            sphere_accs[sphere].append(acc[0])
            
        for sphere, accs in sphere_accs.items():
            if len(accs) > 0:
                result_df[f"Sphere macro: {sphere}"] = sum(accs) / len(accs)
        
        # 整体指标
        if not result_df.empty:
            result_df["Overall macro"] = result_df.mean(axis=1)
            result_df["Overall micro"] = micro_metric["correct"] / micro_metric["total"] if micro_metric["total"] > 0 else 0
        
        # 保存结果
        suffix = eval_file.split(".")[-1]
        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(result_df, score_file)
        
        return result_df