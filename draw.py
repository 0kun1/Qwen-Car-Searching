import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


prediction_path = "output/predictions/sft_test_predictions_transformers.json"
output_dir = Path("output/display_images")
output_dir.mkdir(parents=True, exist_ok=True)


def draw_one(image_path, gt, pred, index):
    img_path = Path("data") / image_path
    img = Image.open(img_path).convert("RGB")

    img_w, img_h = img.size
    scale_w = img_w / 1000.0
    scale_h = img_h / 1000.0

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    # 绿色画 gt
    for item in gt:
        text = item.get("text_content", "")
        bbox = item.get("bbox_2d", [])

        if len(bbox) != 4:
            continue

        x1 = int(bbox[0] * scale_w)
        y1 = int(bbox[1] * scale_h)
        x2 = int(bbox[2] * scale_w)
        y2 = int(bbox[3] * scale_h)

        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        draw.text((x1, max(0, y1 - 28)), f"GT:{text}", fill=(0, 255, 0), font=font)

    # 红色画 pred
    for item in pred:
        text = item.get("text_content", "")
        bbox = item.get("bbox_2d", [])

        if len(bbox) != 4:
            continue

        x1 = int(bbox[0] * scale_w)
        y1 = int(bbox[1] * scale_h)
        x2 = int(bbox[2] * scale_w)
        y2 = int(bbox[3] * scale_h)

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.text((x1, min(img_h - 28, y2 + 5)), f"PRED:{text}", fill=(255, 0, 0), font=font)

    save_name = f"annotated_{index:04d}_{Path(image_path).name}"
    save_path = output_dir / save_name
    img.save(save_path)

    print(f"saved: {save_path}")


def main():
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 你想看的图片编号，可以自己改
    show_index = [0, 1, 2, 3, 4, 10, 20, 50, 100, 200, 300, 400]

    for idx in show_index:
        if idx >= len(data):
            continue

        item = data[idx]

        draw_one(
            image_path=item["image_path"],
            gt=item.get("gt", []),
            pred=item.get("pred", []),
            index=idx,
        )


if __name__ == "__main__":
    main()