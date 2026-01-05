import re
import json
import os.path as osp
import pandas as pd
from ..smp import *
from .image_base import ImageBaseDataset

class SATBench(ImageBaseDataset):
    TYPE = "MCQ"

    def __init__(
        self,
        dataset="SATBench",
        data_file="SATBench",
        data_root=None,
        skip_noimg=True,
    ):
        self.dataset_name = dataset
        self.data_file = data_file
        self.data_root = data_root

        # 加载并预处理数据
        data = self.load_data(data_file)

        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            # 过滤掉没有图片路径的行 (假设 image_paths 列存在)
            data = data[~pd.isna(data["image_paths"])]

        self.meta_only = True
        
        # 确保有 index 列
        if "id" in data.columns:
            data["index"] = data["id"]
        else:
            data["index"] = [str(x) for x in range(len(data))]

        self.data = data
        self.post_build(self.dataset_name)

    def load_data(self, data_file):
        """
        加载数据并进行关键的格式转换：
        1. 将 'answer' (内容字符串) 转为 'A', 'B', 'C'...
        2. 将 'options' 列表格式化并追加到 'question' 文本中
        """
        def load_jsonl(f):
            lines = open(f, encoding="utf-8").readlines()
            lines = [x.strip() for x in lines]
            if lines and lines[-1] == "":
                lines = lines[:-1]
            data = [json.loads(x) for x in lines]
            return pd.DataFrame(data)

        df = load_jsonl(data_file)
        
        # --- 数据预处理核心逻辑 ---
        processed_rows = []
        for idx, row in df.iterrows():
            options = row.get('options', [])
            answer_raw = row.get('answer', '')
            question = row.get('question', '')
            
            # 1. 确定正确选项的字母 (Label Construction)
            label = None
            if isinstance(answer_raw, str) and answer_raw in options:
                # 如果 answer 是内容字符串 (e.g., "Cat")
                ans_idx = options.index(answer_raw)
                label = chr(65 + ans_idx) # 0->A, 1->B...
            elif 'answer_idx' in row:
                # 如果有显式的 index (e.g. circular eval 数据)
                label = chr(65 + int(row['answer_idx']))
            else:
                # 兜底：假设 answer 已经是 "A" 或 "0"
                if str(answer_raw).upper() in ['A', 'B', 'C', 'D', 'E']:
                    label = str(answer_raw).upper()
                else:
                    # 无法确定 Label，标记警告或跳过
                    print(f"Warning: Cannot determine label for ID {row.get('id', idx)}")
                    label = ""

            # 2. 格式化 Prompt (Option Formatting)
            # 构建如: "Question?\n(A) Option1\n(B) Option2..."
            option_str = ""
            for i, opt in enumerate(options):
                option_str += f"({chr(65+i)}) {opt}\n"
            
            # 将选项拼接到问题后，这是 VLM MCQ 的标准做法
            full_question_text = f"{question}\n{option_str}Answer:"

            # 更新 row
            row['question'] = full_question_text # 替换为带选项的完整文本
            row['answer'] = label                # 替换为字母 Label (用于 Eval)
            row['options_formatted'] = option_str # 保留备查
            
            processed_rows.append(row)

        return pd.DataFrame(processed_rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_paths"])
            tgt_path = [osp.join(self.data_root, p) for p in tgt_path]
        else:
            tgt_path = self.dump_image(line)

        # 注意：这里的 question 已经在 load_data 中包含了选项文本
        question = line["question"]
        
        # 兼容处理：如果没有 visual_indices，默认图片在最前
        visual_indices = line.get("visual_indices", [0] * len(tgt_path))

        pil_images = tgt_path
        image_index_pairs = list(zip(pil_images, visual_indices))
        image_index_pairs.sort(key=lambda x: x[1])

        contents = []

        # 简化版逻辑：绝大多数情况图片在最前 (visual_indices 全 0)
        # 如果需要处理 interleaved，这部分逻辑与 ERQABench 保持一致
        if len(visual_indices) == 0 or all(idx == 0 for idx in visual_indices):
            for img, _ in image_index_pairs:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))
        else:
            # Interleaved 逻辑 (保留 ERQABench 原有逻辑)
            # 这里简单处理：直接按顺序插入
            # 实际生产中如果 visual_indices 很复杂，建议使用原始 ERQABench 的详细切分逻辑
            # 但针对 SAT/MCQ 任务，通常是 Image(s) + Text
            for img, _ in image_index_pairs:
                contents.append(dict(type="image", value=img))
            contents.append(dict(type="text", value=question))

        return contents

    @staticmethod
    def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]):
        # 直接复用 ERQABench 的正则提取逻辑，因为它非常健壮
        if isinstance(s, dict):
            s = ""
        s = str(s).strip()
        
        prefixes = [
            r"the (best|correct) answer is",
            r"the answer (is|option is)",
            r"answer:",
            r"option:"
        ]
        for p in prefixes:
            s = re.sub(p, "", s, flags=re.IGNORECASE)

        patterns = [
            r"[\(\[【]([A-G])[\)\]】]",
            r"(?:^|\s)([A-G])(?=[\.:]|$|\s|,)",
            r"\b([A-G])\b"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, s, flags=re.IGNORECASE)
            if matches:
                unique_matches = sorted(list(set(m.upper() for m in matches)))
                return "".join(unique_matches)

        if len(s) < 5:
            fallback = re.findall(r"[A-G]", s, flags=re.IGNORECASE)
            if fallback:
                return "".join(sorted(list(set(m.upper() for m in fallback))))

        return ""

    def evaluate(self, eval_file, **judge_kwargs):
        # 加载推理结果
        data = load(eval_file)
        data["prediction"] = [str(x) for x in data["prediction"]]
        
        # 支持 Circular Eval 的逻辑扩展
        is_circular = "circular_group_id" in data.columns
        
        if is_circular:
            return self._evaluate_circular(data, eval_file)
        else:
            return self._evaluate_standard(data, eval_file)

    def _evaluate_standard(self, data, eval_file):
        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}
        
        for index, it in data.iterrows():
            task = it.get("question_type", "default")
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1
            
            # 提取预测结果
            pred = self.extract_characters_regex(it["prediction"])
            
            # 比对：此时 it["answer"] 已经是 "A", "B" 等字母
            if set(pred) == set(it["answer"]):
                task_stats[task]["correct"] += 1
                micro_metric["correct"] += 1
        
        return self._generate_report(task_stats, micro_metric, eval_file)

    def _evaluate_circular(self, data, eval_file):
        """
        Circular Eval 专用评测逻辑：
        只有当同一个 group_id 下的所有样本都预测正确时，才计分。
        """
        print("Detected Circular Eval format. Calculating consistency score...")
        
        # 提取预测并判断单条是否正确
        data['pred_letter'] = data['prediction'].apply(self.extract_characters_regex)
        data['is_correct'] = data.apply(lambda row: set(row['pred_letter']) == set(row['answer']), axis=1)
        
        # 按 group_id 聚合
        grouped = data.groupby('circular_group_id')
        
        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}
        
        for group_id, group in grouped:
            # 获取该组的任务类型（取第一条即可）
            task = group.iloc[0].get("question_type", "default")
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1
            
            # Circular 核心逻辑：全对才算对
            if group['is_correct'].all():
                task_stats[task]["correct"] += 1
                micro_metric["correct"] += 1

        return self._generate_report(task_stats, micro_metric, eval_file)

    def _generate_report(self, task_stats, micro_metric, eval_file):
        accuracy_dict = {
            task: [stats["correct"] / stats["total"]] for task, stats in sorted(task_stats.items())
        }
        result_df = pd.DataFrame(accuracy_dict)
        
        result_df["Overall micro"] = micro_metric["correct"] / micro_metric["total"]
        
        # 保存结果
        suffix = eval_file.split(".")[-1]
        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(result_df, score_file)
        
        return result_df