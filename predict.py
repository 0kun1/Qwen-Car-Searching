# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor


try:
    from transformers import Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForImageTextToText
    MODEL_CLASS = AutoModelForImageTextToText


DEFAULT_TASKS = [
    {
        "name": "train_correct",
        "input_file": "data/train_sft_correct.json",
        "output_file": "output/predictions/sft_train_correct_enable_vit_predictions.json",
    },
    {
        "name": "train_cut_correct",
        "input_file": "data/train_sft_cut_correct.json",
        "output_file": "output/predictions/sft_train_cut_correct_enable_vit_predictions.json",
    },
    {
        "name": "train_perspective_correct",
        "input_file": "data/train_sft_perspective_correct.json",
        "output_file": "output/predictions/sft_train_perspective_correct_enable_vit_predictions.json",
    },
]


DEFAULT_PROMPT = (
    "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。"
    "输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, "
    "{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"
)


def extract_json_from_response(text):
    """
    从模型输出中提取 JSON list。

    理想输出：
    [{"text_content": "168", "bbox_2d": [833, 828, 962, 884]}]
    """

    if text is None:
        return []

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    return []


def normalize_items(items):
    """
    统一 gt / pred 格式。

    输出格式：
    [
      {
        "text_content": "168",
        "bbox_2d": [833, 828, 962, 884]
      }
    ]
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
            "bbox_2d": bbox,
        })

    return results


def load_gt_from_item(item):
    """
    从 SFT 数据中读取 GT。

    支持两种格式：
    1. SFT 格式：
       item["conversations"][1]["value"]

    2. prediction 格式：
       item["gt"]
    """

    if "conversations" in item:
        try:
            label_str = item["conversations"][1]["value"]
            gt_items = json.loads(label_str)
            return normalize_items(gt_items)
        except Exception:
            return []

    if "gt" in item:
        return normalize_items(item.get("gt", []))

    return []


def get_prompt_from_item(item):
    """
    从 SFT 数据里读取 prompt。
    如果没有 conversations，就使用默认 prompt。
    """

    if "conversations" in item:
        try:
            return item["conversations"][0]["value"]
        except Exception:
            return DEFAULT_PROMPT

    return DEFAULT_PROMPT


def get_image_rel_path(item):
    """
    从数据 item 中读取图片相对路径。

    支持：
    1. SFT 格式：images[0]
    2. prediction 格式：image_path
    """

    if "images" in item:
        return item["images"][0]

    if "image_path" in item:
        return item["image_path"]

    raise ValueError("item 中找不到 images 或 image_path 字段。")


def build_messages(image_abs_path, prompt):
    """
    构造 Qwen3-VL messages。

    注意：
    这里不要用 file://xxx。
    直接传本地绝对路径：
    /root/autodl-tmp/car_search/data/train_images/train_0000.jpg
    """

    prompt = prompt.replace("<image>", "").strip()
    image_abs_path = str(Path(image_abs_path).resolve())

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_abs_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    return messages


def move_inputs_to_device(inputs, device):
    """
    把 processor 输出移动到模型所在设备。
    """

    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    return inputs


@torch.inference_mode()
def infer_one(model, processor, image_abs_path, prompt, max_new_tokens):
    """
    单张图片推理。
    """

    messages = build_messages(
        image_abs_path=image_abs_path,
        prompt=prompt,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs.pop("token_type_ids", None)

    device = next(model.parameters()).device
    inputs = move_inputs_to_device(inputs, device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    pred_items = extract_json_from_response(output_text)
    pred_items = normalize_items(pred_items)

    return pred_items, output_text


def save_json(data, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def predict_one_file(
    model,
    processor,
    input_file,
    output_file,
    data_dir,
    max_new_tokens,
    save_every,
    resume,
):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        print(f"[跳过] 输入文件不存在: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = []
    start_idx = 0

    if resume and output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)

            start_idx = len(all_results)
            print(f"[断点续跑] {output_file} 已有 {start_idx} 条，从第 {start_idx} 条继续。")

        except Exception:
            print(f"[警告] 读取旧输出失败，将从头开始: {output_file}")
            all_results = []
            start_idx = 0

    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总样本数: {len(data)}")
    print(f"开始位置: {start_idx}")
    print("=" * 80)

    for idx in tqdm(range(start_idx, len(data)), desc=f"Predict {input_file.name}"):
        item = data[idx]

        image_rel_path = get_image_rel_path(item)
        image_abs_path = Path(data_dir) / image_rel_path
        image_abs_path = image_abs_path.resolve()

        prompt = get_prompt_from_item(item)
        gt_items = load_gt_from_item(item)

        if not image_abs_path.exists():
            result = {
                "index": idx,
                "image_path": image_rel_path,
                "gt": gt_items,
                "raw_pred": "",
                "pred": [],
                "error": f"image not found: {image_abs_path}",
            }
            all_results.append(result)
            continue

        try:
            pred_items, raw_response = infer_one(
                model=model,
                processor=processor,
                image_abs_path=image_abs_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

            result = {
                "index": idx,
                "image_path": image_rel_path,
                "gt": gt_items,
                "raw_pred": raw_response,
                "pred": pred_items,
                "error": None,
            }

        except Exception as e:
            result = {
                "index": idx,
                "image_path": image_rel_path,
                "gt": gt_items,
                "raw_pred": "",
                "pred": [],
                "error": str(e),
            }

        all_results.append(result)

        if (idx + 1) % save_every == 0:
            save_json(all_results, output_file)

    save_json(all_results, output_file)

    print()
    print(f"保存完成: {output_file}")
    print(f"总处理图片数: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径，例如 saves/qwen3-vl-dpo-merged-aug-4090",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="数据根目录，图片路径会用 data_dir + image_path 拼接。",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="custom",
        choices=[
            "custom",
            "all",
            "train_correct",
            "train_cut_correct",
            "train_perspective_correct",
        ],
        help="选择默认任务。custom 表示使用 --input_file 和 --output_file。",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="自定义输入文件，例如 data/test_sft.json",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="自定义输出文件，例如 output/predictions/dpo_test_predictions_transformers.json",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑。如果之前输出文件有错误结果，先不要开这个。",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("加载模型")
    print("=" * 80)
    print(f"model_path: {args.model_path}")
    print(f"model_class: {MODEL_CLASS}")

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

    model = MODEL_CLASS.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    model.eval()

    if args.task == "custom":
        if args.input_file is None or args.output_file is None:
            raise ValueError("task=custom 时必须提供 --input_file 和 --output_file")

        tasks = [
            {
                "name": "custom",
                "input_file": args.input_file,
                "output_file": args.output_file,
            }
        ]

    elif args.task == "all":
        tasks = DEFAULT_TASKS

    else:
        tasks = [x for x in DEFAULT_TASKS if x["name"] == args.task]

    for task in tasks:
        print()
        print(f"开始任务: {task['name']}")

        predict_one_file(
            model=model,
            processor=processor,
            input_file=task["input_file"],
            output_file=task["output_file"],
            data_dir=args.data_dir,
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
            resume=args.resume,
        )

    print()
    print("=" * 80)
    print("全部预测完成")
    print("=" * 80)


if __name__ == "__main__":
    main()