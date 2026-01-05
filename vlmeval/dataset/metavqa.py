import re

import pandas as pd

from ..smp import *
from .image_base import ImageBaseDataset


class MetaVQABench(ImageBaseDataset):
    TYPE = "MCQ"

    def __init__(
        self,
        dataset="MetaVQABench",
        data_file="MetaVQABench",
        data_root=None,
        skip_noimg=True,
    ):
        # You can override this variable to save image files to a different directory
        self.dataset_name = "MetaVQABench"
        self.data_file = data_file
        self.data_root = data_root

        data = self.load_data(data_file)


        self.skip_noimg = skip_noimg
        if skip_noimg and "image" in data:
            data = data[~pd.isna(data["image"])]

        self.meta_only = True

        # the dataframe has `id` field, which is the index
        data["index"] = data["id"]

        self.data = data
        self.post_build(self.dataset_name)

    def load_data(self, data_file="MetaVQABench"):
        def load_jsonl(f):
            lines = open(f, encoding="utf-8").readlines()
            lines = [x.strip() for x in lines]
            if lines[-1] == "":
                lines = lines[:-1]
            data = [json.loads(x) for x in lines]
            return pd.DataFrame(data)

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
        visual_indices = line["visual_indices"]

        # Convert encoded images to PIL images
        pil_images = tgt_path

        # Prepare contents for API based on visual_indices
        # Create a list of (image, index) pairs
        image_index_pairs = list(zip(pil_images, visual_indices, strict=False))

        # Sort by visual_indices
        image_index_pairs.sort(key=lambda x: x[1])

        # Split the question text and interleave with images
        contents = []

        # Handle case where visual_indices is empty (place images at the beginning)
        if len(visual_indices) == 0:
            # Add all images at the beginning
            for img in pil_images:
                contents.append(dict(type="image", value=img))
            # Then add the question text
            contents.append(dict(type="text", value=question))
        # Handle case where all indices are 0 (all images at the beginning)
        elif all(idx == 0 for idx in visual_indices):
            # First add all images
            for img, _ in image_index_pairs:
                contents.append(dict(type="image", value=img))
            # Then add the question text
            contents.append(dict(type="text", value=question))
        else:
            # Split question at visual_indices positions
            last_pos = 0

            # Process each image and its position
            for img, idx in image_index_pairs:
                # if idx == 0:
                    # Image goes at the beginning
                contents.append(dict(type="image", value=img))
                # else:
                #     # Add text segment before this image
                #     if idx <= len(question):
                #         text_segment = question[last_pos:idx]
                #         if text_segment:
                #             contents.append(dict(type="text", value=text_segment))
                #         contents.append(dict(type="image", value=img))
                #         last_pos = idx
                #     else:
                #         # If index is beyond question length, just append the image
                #         contents.append(dict(type="image", value=img))

            # Add any remaining text
            if last_pos < len(question):
                contents.append(dict(type="text", value=question[last_pos:]))

            # If no content was added (e.g., all indices were beyond question length),
            # add the full question at the beginning
            if not contents:
                contents.append(dict(type="text", value=question))
                for img, _ in image_index_pairs:
                    contents.append(dict(type="image", value=img))

        # print(contents)
        return contents

    @staticmethod
    def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]):
        if isinstance(s, dict):
            s = ""
        s = str(s).strip() # 强制转字符串并去空
        
        # 1. 预处理：移除无用的前缀文本（忽略大小写）
        # 使用正则替换，避免大小写不匹配问题
        prefixes = [
            r"the (best|correct) answer is",
            r"the answer (is|option is)",
            r"answer:",
            r"option:"
        ]
        for p in prefixes:
            s = re.sub(p, "", s, flags=re.IGNORECASE)

        # 2. 定义匹配模式（优先级从高到低）
        # 这里的关键是使用 \b (单词边界) 和明确的标点符号
        patterns = [
            # Pattern 1: 括号包裹，如 (A), [A], 【A】
            r"[\(\[【]([A-G])[\)\]】]",
            
            # Pattern 2: 字母+点/冒号，位于行首或空格后，如 "A." "A:" "Answer: A."
            # (?<=^|\s) 确保前面是行首或空格
            # ([A-G]) 捕获字母
            # (?=[\.:\s]|$|,) 确保后面跟着点、冒号、空格、逗号或行尾
            r"(?:^|\s)([A-G])(?=[\.:]|$|\s|,)",
            
            # Pattern 3: 严格的孤立字母（仅当它前后都是边界时）
            r"\b([A-G])\b"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, s, flags=re.IGNORECASE)
            if matches:
                # 找到匹配后立即去重并返回，不再执行后续更宽松的规则
                # 使用 set 去重，sorted 排序确保顺序一致 (如 "A" 而不是随机顺序)
                unique_matches = sorted(list(set(m.upper() for m in matches)))
                return "".join(unique_matches)

        # 3. 兜底逻辑：只有当字符串极短（例如就是 "A" 或 "A."）且上面没匹配到时才允许
        # 防止长句子中的单词被误拆解
        if len(s) < 5:
            fallback = re.findall(r"[A-G]", s, flags=re.IGNORECASE)
            if fallback:
                return "".join(sorted(list(set(m.upper() for m in fallback))))

        return ""

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data["prediction"] = [str(x) for x in data["prediction"]]
        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}
        for index, it in data.iterrows():
            task = it["question_type"]
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1
            pred = self.extract_characters_regex(it["prediction"])
            if set(pred) == set(it["answer"]):
                task_stats[task]["correct"] += 1
                micro_metric["correct"] += 1
        accuracy_dict = {
            task: [stats["correct"] / stats["total"]] for task, stats in sorted(task_stats.items())
        }
        result_df = pd.DataFrame(accuracy_dict)
        from collections import defaultdict

        sphere_accs = defaultdict(list)
        for task, acc in accuracy_dict.items():
            sphere = task.split("/")[0]
            assert len(acc) == 1
            sphere_accs[sphere].append(acc[0])
        for sphere, accs in sphere_accs.items():
            result_df[f"Sphere macro: {sphere}"] = sum(accs) / len(accs)
        result_df["Overall macro"] = result_df.mean(axis=1)
        result_df["Overall micro"] = micro_metric["correct"] / micro_metric["total"]
        suffix = eval_file.split(".")[-1]
        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(result_df, score_file)
        return result_df
