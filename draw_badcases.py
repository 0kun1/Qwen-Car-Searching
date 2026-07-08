import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size=24):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    for p in font_paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)

    return ImageFont.load_default()


def scale_bbox(bbox, img_w, img_h):
    scale_w = img_w / 1000.0
    scale_h = img_h / 1000.0

    x1 = int(bbox[0] * scale_w)
    y1 = int(bbox[1] * scale_h)
    x2 = int(bbox[2] * scale_w)
    y2 = int(bbox[3] * scale_h)

    return [x1, y1, x2, y2]


def draw_label(draw, xy, text, fill, font):
    x, y = xy

    try:
        bbox = draw.textbbox((x, y), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = 500, 28

    draw.rectangle(
        [x, y, x + text_w + 10, y + text_h + 10],
        fill=(0, 0, 0),
    )

    draw.text(
        (x + 5, y + 5),
        text,
        fill=fill,
        font=font,
    )


def get_main_error_type(errors):
    """
    一张图可能有多个错误。
    优先级：
    text_error > missing > hallucination
    """
    types = [e.get("type") for e in errors]

    if "text_error" in types:
        return "text_error"
    elif "missing" in types:
        return "missing"
    elif "hallucination" in types:
        return "hallucination"
    else:
        return "other"


def draw_one_badcase(image_path, errors, output_path):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    draw = ImageDraw.Draw(img)
    font = load_font(24)

    y_offset = 20

    for err in errors:
        err_type = err.get("type")

        # 1. Text error: bbox matched, but text is wrong
        if err_type == "text_error":
            gt_text = err.get("gt", "")
            pred_text = err.get("pred", "")
            error_type = err.get("error_type", "")
            details = err.get("details", "")
            iou = err.get("iou", 0)

            gt_bbox = scale_bbox(err["gt_bbox"], img_w, img_h)
            pred_bbox = scale_bbox(err["pred_bbox"], img_w, img_h)

            # GT: green
            draw.rectangle(gt_bbox, outline=(0, 255, 0), width=5)
            draw_label(
                draw,
                (gt_bbox[0], max(0, gt_bbox[1] - 36)),
                f"GT: {gt_text}",
                (0, 255, 0),
                font,
            )

            # PRED: red
            draw.rectangle(pred_bbox, outline=(255, 0, 0), width=5)
            draw_label(
                draw,
                (pred_bbox[0], min(img_h - 36, pred_bbox[3] + 6)),
                f"PRED: {pred_text}",
                (255, 0, 0),
                font,
            )

            # English error message, avoid Chinese garbled text
            draw_label(
                draw,
                (20, y_offset),
                f"TEXT_ERROR | {error_type} | IoU={iou:.3f} | {details}",
                (255, 255, 0),
                font,
            )

            y_offset += 42

        # 2. Missing: GT exists, but model missed it
        elif err_type == "missing":
            gt_text = err.get("gt", "")
            gt_bbox = scale_bbox(err["gt_bbox"], img_w, img_h)

            # Missing GT: blue
            draw.rectangle(gt_bbox, outline=(0, 128, 255), width=6)
            draw_label(
                draw,
                (gt_bbox[0], max(0, gt_bbox[1] - 36)),
                f"MISSING GT: {gt_text}",
                (0, 128, 255),
                font,
            )

            draw_label(
                draw,
                (20, y_offset),
                f"MISSING | GT: {gt_text}",
                (0, 128, 255),
                font,
            )

            y_offset += 42

        # 3. Hallucination: model predicts non-existing parking number
        elif err_type == "hallucination":
            pred_text = err.get("pred", "")
            pred_bbox = scale_bbox(err["pred_bbox"], img_w, img_h)

            # Hallucination PRED: red
            draw.rectangle(pred_bbox, outline=(255, 0, 0), width=6)
            draw_label(
                draw,
                (pred_bbox[0], max(0, pred_bbox[1] - 36)),
                f"FAKE PRED: {pred_text}",
                (255, 0, 0),
                font,
            )

            draw_label(
                draw,
                (20, y_offset),
                f"HALLUCINATION | PRED: {pred_text}",
                (255, 0, 0),
                font,
            )

            y_offset += 42

    img.save(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--error_file",
        type=str,
        default="output/analysis/sft_aug_test_transformers_predictions_errors.json",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/test_images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/badcase_images_aug",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "text_error", "missing", "hallucination"],
    )

    args = parser.parse_args()

    error_file = Path(args.error_file)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    with open(error_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    errors_by_image = report["errors_by_image"]

    count = 0

    for image_name, errors in errors_by_image.items():
        if args.only != "all":
            errors = [e for e in errors if e.get("type") == args.only]

            if len(errors) == 0:
                continue

        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"image not found, skip: {image_path}")
            continue

        main_error_type = get_main_error_type(errors)

        # 按错误类型分文件夹
        save_dir = output_dir / main_error_type
        save_dir.mkdir(parents=True, exist_ok=True)

        save_name = f"badcase_{count:04d}_{image_name}"
        output_path = save_dir / save_name

        draw_one_badcase(
            image_path=image_path,
            errors=errors,
            output_path=output_path,
        )

        print(f"saved: {output_path}")

        count += 1

        if args.limit is not None and count >= args.limit:
            break

    print(f"\nDone. Saved {count} badcase images.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()