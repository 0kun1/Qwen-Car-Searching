# -*- coding: utf-8 -*-

import argparse
import json
import os


PROMPT = (
    '<image>你是一个专业的车位号识别助手。'
    '请观察图片，输出车位号的位置与编号。'
    '输出json格式：'
    '[{"text_content": text, "bbox_2d": [x1, y1, x2, y2]}, '
    '{"text_content": text, "bbox_2d": [x1, y1, x2, y2]}]'
)


def round_coord(x):
    """
    把 bbox 坐标转成整数。
    原始标注里可能是浮点数，比如 833.2909。
    训练输出里一般希望是整数坐标。
    """
    return int(round(float(x)))


def convert_one_item(item):
    """
    把一条原始样本转换成一条 SFT 样本。
    """

    answer_list = item.get("answer", [])
    parking_slots = []

    for ans in answer_list:
        label = ans.get("label")

        # 原始数据中可能用 label == "无" 表示没有车位号
        if label == "无":
            continue

        bbox = ans.get("bbox")
        if not bbox:
            continue

        xmin = round_coord(bbox["xmin"])
        ymin = round_coord(bbox["ymin"])
        xmax = round_coord(bbox["xmax"])
        ymax = round_coord(bbox["ymax"])

        if not (0 <= xmin < xmax <= 1000 and 0 <= ymin < ymax <= 1000):
            print(f"跳过非法 bbox: label={label}, bbox={[xmin, ymin, xmax, ymax]}, image={item['new_img']}")
            continue

        parking_slots.append({
            "text_content": str(label),
            "bbox_2d": [xmin, ymin, xmax, ymax]
        })

    sft_item = {
        "conversations": [
            {
                "from": "human",
                "value": PROMPT
            },
            {
                "from": "gpt",
                "value": json.dumps(parking_slots, ensure_ascii=False)
            }
        ],
        "images": [item["new_img"]]
    }

    return sft_item


def convert_dataset(input_path, output_path):
    """
    把整个原始 json 文件转换成 SFT json 文件。
    """

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    sft_data = []

    for item in raw_data:
        sft_item = convert_one_item(item)
        sft_data.append(sft_item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成：{input_path} -> {output_path}")
    print(f"样本数量：{len(sft_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    train_input = os.path.join(args.data_dir, "train_labels_with_bbox.json")
    train_output = os.path.join(args.data_dir, "train_sft.json")

    test_input = os.path.join(args.data_dir, "test_labels_with_bbox.json")
    test_output = os.path.join(args.data_dir, "test_sft.json")

    convert_dataset(train_input, train_output)
    convert_dataset(test_input, test_output)


if __name__ == "__main__":
    main()