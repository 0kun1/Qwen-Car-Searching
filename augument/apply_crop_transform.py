# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 随机crop增强
# --------------------------------------------

# 1. crop 之后车位号可能减少
# 2. crop 之后图片不合理，丢弃
# 3. bbox、text 都需要做相应调整

import json
import os
import random
import numpy as np
from PIL import Image
import albumentations as A


def parse_conversations(value):
    """
    从 conversations 的 gpt value 中解析出标注数据。

    原始 value 是字符串形式：
    "[{\"text_content\":\"168\",\"bbox_2d\":[833,828,962,884]}]"

    解析后变成 Python list：
    [
        {
            "text_content": "168",
            "bbox_2d": [833, 828, 962, 884]
        }
    ]
    """
    try:
        return json.loads(value)
    except:
        return []


def get_safe_crop_params(annotations, orig_w, orig_h, min_crop_ratio=0.6):
    """
    针对单框情况：
    计算一个包含该 bbox 的安全裁剪区域。

    为什么单目标要特殊处理？
    因为一张图只有一个车位号，如果随机 crop 把它裁没了，
    这张图就变成无效训练样本了。

    所以这里会围绕 bbox 中心点裁剪，尽量保证目标还在图中。
    """
    ann = annotations[0]
    bbox = ann["bbox_2d"]

    # 原始 bbox 是 0-1000 坐标
    # 这里先转成原图像素坐标
    x1 = bbox[0] * orig_w / 1000
    y1 = bbox[1] * orig_h / 1000
    x2 = bbox[2] * orig_w / 1000
    y2 = bbox[3] * orig_h / 1000

    bw = x2 - x1
    bh = y2 - y1

    # 随机确定 crop 尺寸，范围是原图的 60% 到 95%
    scale = random.uniform(min_crop_ratio, 0.95)
    crop_w = orig_w * scale
    crop_h = orig_h * scale

    # 如果目标框本身比 crop 区域还大，就扩大 crop 区域
    if bw > crop_w or bh > crop_h:
        crop_w = min(orig_w, bw * 1.2)
        crop_h = min(orig_h, bh * 1.2)

    # bbox 中心点
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 根据中心点确定 crop 左上角
    x_min = int(max(0, min(orig_w - crop_w, cx - crop_w / 2)))
    y_min = int(max(0, min(orig_h - crop_h, cy - crop_h / 2)))

    x_max = int(x_min + crop_w)
    y_max = int(y_min + crop_h)

    return x_min, y_min, x_max, y_max


def process_image(item, output_image_dir, output_idx, max_retries=15):
    """
    处理单张图片：
    1. 读取图片
    2. 解析 bbox 和 text
    3. 做随机 crop
    4. 同步更新 bbox
    5. 保存增强后的图片
    6. 返回新图片路径和新标注
    """
    image_path = item["images"][0]
    full_image_path = os.path.join("data", image_path)

    if not os.path.exists(full_image_path):
        print(f"警告: 找不到图片 {image_path}")
        return None, None

    img = Image.open(full_image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)

    # 解析 gpt 标注
    annotations = parse_conversations(item["conversations"][1]["value"])

    # 分离 bbox 和文字
    bboxes = []
    text_labels = []

    for ann in annotations:
        b = ann["bbox_2d"]

        # albumentations 的 bbox 需要 0-1 归一化坐标
        bboxes.append([
            b[0] / 1000,
            b[1] / 1000,
            b[2] / 1000,
            b[3] / 1000,
        ])

        # 文字单独保存
        text_labels.append(ann["text_content"])

    # bbox_params 负责告诉 albumentations：
    # 1. bbox 格式是 albumentations，也就是 0-1 坐标
    # 2. bbox 至少保留 95% 面积才算有效
    # 3. text_labels 和 bbox 是一一对应的
    bbox_params = A.BboxParams(
        format="albumentations",
        min_visibility=0.95,
        label_fields=["text_labels"],
    )

    final_img = None
    final_bboxes = []
    final_labels = []

    # 多次尝试 crop，避免生成无效样本
    for _ in range(max_retries):

        # 单目标图片：安全 crop
        if len(annotations) == 1:
            x_min, y_min, x_max, y_max = get_safe_crop_params(
                annotations,
                orig_w,
                orig_h,
            )

            aug = A.Compose(
                [
                    A.Crop(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )
                ],
                bbox_params=bbox_params,
            )

        # 多目标图片：随机 crop
        else:
            scale = random.uniform(0.6, 0.9)

            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)

            aug = A.Compose(
                [
                    A.RandomCrop(
                        height=new_h,
                        width=new_w,
                    )
                ],
                bbox_params=bbox_params,
            )

        # 同时传入 image、bboxes、text_labels
        transformed = aug(
            image=img_np,
            bboxes=bboxes,
            text_labels=text_labels,
        )

        # 如果 crop 后仍然有 bbox，或者原图本来就是空标注，则保留
        if len(transformed["bboxes"]) > 0 or len(annotations) == 0:
            final_img = transformed["image"]
            final_bboxes = transformed["bboxes"]
            final_labels = transformed["text_labels"]
            break

    # 如果多次 crop 都失败，就退回原图，保证不丢数据
    if final_img is None:
        final_img = img_np
        final_bboxes = bboxes
        final_labels = text_labels

    # 重新构造增强后的标注
    new_annotations = []

    for b, label in zip(final_bboxes, final_labels):
        new_annotations.append({
            "text_content": label,
            "bbox_2d": [
                max(0, min(1000, int(round(b[0] * 1000)))),
                max(0, min(1000, int(round(b[1] * 1000)))),
                max(0, min(1000, int(round(b[2] * 1000)))),
                max(0, min(1000, int(round(b[3] * 1000)))),
            ],
        })

    # 保存增强后的图片
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    new_filename = f"{name}_cut_{output_idx}{ext}"
    output_path = os.path.join(output_image_dir, new_filename)

    Image.fromarray(final_img).save(output_path, quality=95)

    # 写入 SFT json 里的图片路径
    container_img_path = os.path.join("train_images_cut", new_filename)

    return container_img_path, new_annotations


def main():
    random.seed(42)

    input_json = "data/train_sft_correct.json"
    output_json = "data/train_sft_cut_correct.json"
    local_output_dir = "data/train_images_cut"

    os.makedirs(local_output_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    new_dataset = []

    print(f"开始处理，共 {len(dataset)} 条数据...")

    for idx, item in enumerate(dataset):
        try:
            new_path, new_anns = process_image(
                item,
                local_output_dir,
                idx,
            )

            if new_path is None:
                continue

            new_item = {
                "conversations": [
                    item["conversations"][0],
                    {
                        "from": "gpt",
                        "value": json.dumps(new_anns, ensure_ascii=False),
                    },
                ],
                "images": [new_path],
            }

            new_dataset.append(new_item)

            if (idx + 1) % 50 == 0:
                print(f"已处理: {idx + 1}/{len(dataset)}")

        except Exception as e:
            print(f"处理第 {idx} 张图出错: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"处理完成！新数据集已保存至 {output_json}")


if __name__ == "__main__":
    main()