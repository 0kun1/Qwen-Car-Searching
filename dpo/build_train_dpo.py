# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


PROMPT = (
    "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。"
    "输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, "
    "{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"
)


DEFAULT_INPUT_FILES = [
    "output/predictions/sft_train_correct_enable_vit_predictions.json",
    "output/predictions/sft_train_cut_correct_enable_vit_predictions.json",
    "output/predictions/sft_train_perspective_correct_enable_vit_predictions.json",
]


def normalize_items(items):
    """
    清洗 gt / pred，保证格式统一。
    """

    if not isinstance(items, list):
        return []

    results = []

    for item in items:
        if not isinstance(item, dict):
            continue

        text = item.get("text_content")
        bbox = item.get("bbox_2d")

        if text is None or bbox is None:
            continue

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        try:
            bbox = [int(round(float(x))) for x in bbox]
        except Exception:
            continue

        results.append({
            "text_content": str(text),
            "bbox_2d": bbox
        })

    return results


def calculate_bbox_area(bbox):
    """
    计算 bbox 面积。
    """

    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_largest_bbox_text(items):
    """
    找到 pred 里面积最大的 bbox 对应的 text_content。
    老师代码就是用这个逻辑筛明显错误样本。
    """

    items = normalize_items(items)

    if not items:
        return None

    largest = max(
        items,
        key=lambda x: calculate_bbox_area(x["bbox_2d"])
    )

    return largest["text_content"]


def is_hard_negative(item):
    """
    判断是不是明显错误样本。

    老师代码逻辑：
    如果 pred 中面积最大的框对应的文字不在 gt 里，
    就认为这是一个值得加入 DPO 的错误样本。
    """

    gt_list = normalize_items(item.get("gt", []))
    pred_list = normalize_items(item.get("pred", []))

    if not pred_list:
        return False

    largest_pred_text = find_largest_bbox_text(pred_list)

    gt_texts = {
        gt["text_content"]
        for gt in gt_list
    }

    return largest_pred_text not in gt_texts


def is_valid_dpo_pair(gt_list, pred_list):
    """
    判断是否适合作为 DPO pair。

    chosen 和 rejected 完全一样就没必要加入。
    gt 和 pred 都为空也没必要加入。
    """

    if gt_list == pred_list:
        return False

    if len(gt_list) == 0 and len(pred_list) == 0:
        return False

    return True


def build_dpo_item(image_path, gt_list, pred_list):
    """
    构造 LLaMAFactory DPO 所需 sharegpt ranking 格式。
    chosen = gt
    rejected = pred
    """

    return {
        "conversations": [
            {
                "from": "human",
                "value": PROMPT
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": json.dumps(gt_list, ensure_ascii=False)
        },
        "rejected": {
            "from": "gpt",
            "value": json.dumps(pred_list, ensure_ascii=False)
        },
        "images": [
            image_path
        ]
    }


def load_prediction_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_files",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="输入的 predictions.json 文件"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/train_dpo.json",
        help="输出 DPO 文件"
    )

    parser.add_argument(
        "--extra_sample_num",
        type=int,
        default=500,
        help="除了 hard negative 以外，额外抽取多少普通样本"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    random.seed(args.seed)

    hard_candidates = []
    extra_candidates = []

    seen_hard_images = set()
    seen_extra_images = set()

    print("=" * 80)
    print("开始构造 DPO 数据")
    print("=" * 80)

    for file_path in args.input_files:
        path = Path(file_path)

        if not path.exists():
            print(f"[跳过] 文件不存在: {path}")
            continue

        data = load_prediction_file(path)

        print()
        print(f"读取文件: {path}")
        print(f"样本数: {len(data)}")

        file_hard_count = 0
        file_extra_count = 0

        for item in data:
            image_path = item.get("image_path")
            gt_list = normalize_items(item.get("gt", []))
            pred_list = normalize_items(item.get("pred", []))
            error = item.get("error", None)

            if image_path is None:
                continue

            # 推理报错的样本先不加入 DPO
            if error is not None:
                continue

            if not is_valid_dpo_pair(gt_list, pred_list):
                continue

            new_item = {
                "image_path": image_path,
                "gt": gt_list,
                "pred": pred_list
            }

            if is_hard_negative(new_item):
                if image_path not in seen_hard_images:
                    hard_candidates.append(new_item)
                    seen_hard_images.add(image_path)
                    file_hard_count += 1
            else:
                if image_path not in seen_extra_images:
                    extra_candidates.append(new_item)
                    seen_extra_images.add(image_path)
                    file_extra_count += 1

        print(f"明显错误样本 hard candidates: {file_hard_count}")
        print(f"普通补充候选 extra candidates: {file_extra_count}")

    print()
    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"hard candidates 总数: {len(hard_candidates)}")
    print(f"extra candidates 总数: {len(extra_candidates)}")

    # 随机抽取普通补充样本
    sample_num = min(args.extra_sample_num, len(extra_candidates))
    selected_extra = random.sample(extra_candidates, sample_num)

    print(f"抽取 extra 样本数: {len(selected_extra)}")

    final_candidates = hard_candidates + selected_extra

    # 最终按 image_path 去重
    final_dpo = []
    seen_images = set()

    for item in final_candidates:
        image_path = item["image_path"]

        if image_path in seen_images:
            continue

        seen_images.add(image_path)

        dpo_item = build_dpo_item(
            image_path=image_path,
            gt_list=item["gt"],
            pred_list=item["pred"]
        )

        final_dpo.append(dpo_item)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dpo, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("DPO 数据构造完成")
    print("=" * 80)
    print(f"输出文件: {output_path}")
    print(f"最终 DPO 样本数: {len(final_dpo)}")


if __name__ == "__main__":
    main()