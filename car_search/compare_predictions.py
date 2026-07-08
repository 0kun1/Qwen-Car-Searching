# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from collections import Counter
import difflib
import csv


IOU_THRESHOLD = 0.5


def calculate_iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def safe_text(x):
    if x is None:
        return ""
    return str(x).strip()


def normalize_items(items):
    """
    统一 gt / pred 格式。

    兼容：
    {
      "text_content": "168",
      "bbox_2d": [x1, y1, x2, y2]
    }

    输出：
    [
      {
        "text_content": "...",
        "bbox_2d": [...]
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
            bbox = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        results.append({
            "text_content": safe_text(text),
            "bbox_2d": bbox,
        })

    return results


def get_image_key(item):
    """
    不同文件里的 image_path 可能长这样：

    test_images/test_0000.jpg
    ./data/test_images/test_0000.jpg

    为了对齐不同模型，统一只用 basename：
    test_0000.jpg
    """

    image_path = item.get("image_path")

    if image_path is None and "images" in item:
        images = item.get("images", [])
        if images:
            image_path = images[0]

    if image_path is None:
        return ""

    return os.path.basename(str(image_path))


def classify_text_error(gt_text, pred_text):
    """
    粗略分类文本错误：
    - 字符错检
    - 内容缺失
    - 多余字符
    """

    gt_text = safe_text(gt_text)
    pred_text = safe_text(pred_text)

    diff = list(difflib.ndiff(gt_text, pred_text))

    replaced = []
    missing = []
    extra = []

    i = 0
    while i < len(diff):
        tag = diff[i][0]
        char = diff[i][-1]

        if tag == "-":
            # 看下一位是不是 "+"
            if i + 1 < len(diff) and diff[i + 1][0] == "+":
                replaced.append((char, diff[i + 1][-1]))
                i += 2
                continue
            else:
                missing.append(char)

        elif tag == "+":
            extra.append(char)

        i += 1

    error_types = []
    details = []

    if replaced:
        error_types.append("字符错检")
        details.append(
            "字符替换: " + ", ".join([f"{a}->{b}" for a, b in replaced])
        )

    if missing:
        error_types.append("内容缺失")
        details.append("缺失字符: " + "".join(missing))

    if extra:
        error_types.append("多余字符")
        details.append("多余字符: " + "".join(extra))

    if not error_types:
        error_types.append("文本错误")
        details.append(f"{gt_text} -> {pred_text}")

    return "|".join(error_types), "; ".join(details)


def match_predictions(gt_items, pred_items, iou_threshold=IOU_THRESHOLD):
    """
    按 IoU 做一对一匹配。

    返回：
    matched_pairs: [(gt_idx, pred_idx, iou), ...]
    unmatched_gt: [gt_idx, ...]
    unmatched_pred: [pred_idx, ...]
    """

    pairs = []

    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            iou = calculate_iou(gt["bbox_2d"], pred["bbox_2d"])
            if iou >= iou_threshold:
                pairs.append((gi, pi, iou))

    # IoU 从大到小匹配，避免一个 pred 匹配多个 gt
    pairs.sort(key=lambda x: x[2], reverse=True)

    used_gt = set()
    used_pred = set()
    matched_pairs = []

    for gi, pi, iou in pairs:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)
        matched_pairs.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt_items)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_items)) if i not in used_pred]

    return matched_pairs, unmatched_gt, unmatched_pred


def evaluate_one_item(item, iou_threshold=IOU_THRESHOLD):
    """
    评估单张图片。
    """

    gt_items = normalize_items(item.get("gt", []))
    pred_items = normalize_items(item.get("pred", []))

    matched_pairs, unmatched_gt, unmatched_pred = match_predictions(
        gt_items,
        pred_items,
        iou_threshold=iou_threshold,
    )

    correct_text = 0
    wrong_text = 0
    errors = []

    for gi, pi, iou in matched_pairs:
        gt = gt_items[gi]
        pred = pred_items[pi]

        gt_text = safe_text(gt["text_content"])
        pred_text = safe_text(pred["text_content"])

        if gt_text == pred_text:
            correct_text += 1
        else:
            wrong_text += 1
            error_type, details = classify_text_error(gt_text, pred_text)

            errors.append({
                "type": "text_error",
                "error_type": error_type,
                "details": details,
                "gt": gt_text,
                "pred": pred_text,
                "iou": iou,
                "gt_bbox": gt["bbox_2d"],
                "pred_bbox": pred["bbox_2d"],
            })

    for gi in unmatched_gt:
        gt = gt_items[gi]
        errors.append({
            "type": "missing",
            "error_type": "漏检",
            "gt": safe_text(gt["text_content"]),
            "gt_bbox": gt["bbox_2d"],
        })

    for pi in unmatched_pred:
        pred = pred_items[pi]
        errors.append({
            "type": "hallucination",
            "error_type": "无中生有",
            "pred": safe_text(pred["text_content"]),
            "pred_bbox": pred["bbox_2d"],
        })

    tp = correct_text
    fp = len(pred_items) - correct_text
    fn = len(gt_items) - correct_text

    return {
        "image_key": get_image_key(item),
        "num_gt": len(gt_items),
        "num_pred": len(pred_items),
        "num_matched_bbox": len(matched_pairs),
        "sum_iou": sum(x[2] for x in matched_pairs),
        "correct_text": correct_text,
        "wrong_text": wrong_text,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "errors": errors,
    }


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_file(model_name, file_path, iou_threshold=IOU_THRESHOLD):
    """
    评估一个模型的 prediction json 文件。
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_images = len(data)
    total_gt = 0
    total_pred = 0

    matched_bbox = 0
    sum_iou = 0.0

    correct_text = 0
    wrong_text = 0

    tp = 0
    fp = 0
    fn = 0

    inference_error_images = 0
    eval_error_images = 0

    empty_stats = Counter()
    error_type_counts = Counter()
    errors_by_image = {}
    per_image = {}

    for item in data:
        if item.get("error") is not None:
            inference_error_images += 1

        one = evaluate_one_item(item, iou_threshold=iou_threshold)

        image_key = one["image_key"]

        total_gt += one["num_gt"]
        total_pred += one["num_pred"]

        matched_bbox += one["num_matched_bbox"]
        sum_iou += one["sum_iou"]

        correct_text += one["correct_text"]
        wrong_text += one["wrong_text"]

        tp += one["tp"]
        fp += one["fp"]
        fn += one["fn"]

        if one["num_gt"] == 0 and one["num_pred"] == 0:
            empty_stats["gt_empty_pred_empty"] += 1
        elif one["num_gt"] > 0 and one["num_pred"] == 0:
            empty_stats["gt_nonempty_pred_empty"] += 1
        elif one["num_gt"] == 0 and one["num_pred"] > 0:
            empty_stats["gt_empty_pred_nonempty"] += 1
        else:
            empty_stats["gt_nonempty_pred_nonempty"] += 1

        if one["errors"]:
            eval_error_images += 1
            errors_by_image[image_key] = one["errors"]

            for err in one["errors"]:
                error_type_counts[err["error_type"]] += 1

        # 单图 end-to-end F1，用于模型之间逐图对比
        p_img = safe_div(one["tp"], one["tp"] + one["fp"])
        r_img = safe_div(one["tp"], one["tp"] + one["fn"])

        # 如果 GT 和 pred 都为空，认为这张图完全正确，score = 1
        if one["num_gt"] == 0 and one["num_pred"] == 0:
            img_f1 = 1.0
        else:
            img_f1 = f1_score(p_img, r_img)

        per_image[image_key] = {
            "tp": one["tp"],
            "fp": one["fp"],
            "fn": one["fn"],
            "end_to_end_f1": img_f1,
            "num_errors": len(one["errors"]),
        }

    bbox_precision = safe_div(matched_bbox, total_pred)
    bbox_recall = safe_div(matched_bbox, total_gt)
    bbox_f1 = f1_score(bbox_precision, bbox_recall)
    avg_iou = safe_div(sum_iou, matched_bbox)

    text_acc = safe_div(correct_text, matched_bbox)

    e2e_precision = safe_div(tp, tp + fp)
    e2e_recall = safe_div(tp, tp + fn)
    e2e_f1 = f1_score(e2e_precision, e2e_recall)

    summary = {
        "model_name": model_name,
        "file_path": str(file_path),
        "iou_threshold": iou_threshold,
        "basic": {
            "total_images": total_images,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "inference_error_images": inference_error_images,
            "eval_error_images": eval_error_images,
        },
        "empty_stats": dict(empty_stats),
        "bbox_match": {
            "matched": matched_bbox,
            "precision": bbox_precision,
            "recall": bbox_recall,
            "f1": bbox_f1,
            "avg_iou": avg_iou,
        },
        "text_on_matched_bbox": {
            "correct_text": correct_text,
            "wrong_text": wrong_text,
            "accuracy_on_matched": text_acc,
        },
        "end_to_end": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": e2e_precision,
            "recall": e2e_recall,
            "f1": e2e_f1,
        },
        "error_type_counts": dict(error_type_counts),
        "total_errors": sum(error_type_counts.values()),
    }

    return {
        "summary": summary,
        "errors_by_image": errors_by_image,
        "per_image": per_image,
    }


def compare_pairwise(results):
    """
    逐图比较不同模型谁更好。

    用单图 end_to_end_f1 比较。
    """

    model_names = list(results.keys())
    pairwise = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a = model_names[i]
            b = model_names[j]

            images = set(results[a]["per_image"].keys()) | set(results[b]["per_image"].keys())

            a_win = []
            b_win = []
            tie = []

            for image_key in sorted(images):
                a_score = results[a]["per_image"].get(image_key, {}).get("end_to_end_f1", 0.0)
                b_score = results[b]["per_image"].get(image_key, {}).get("end_to_end_f1", 0.0)

                if a_score > b_score:
                    a_win.append({
                        "image": image_key,
                        a: a_score,
                        b: b_score,
                    })
                elif b_score > a_score:
                    b_win.append({
                        "image": image_key,
                        a: a_score,
                        b: b_score,
                    })
                else:
                    tie.append({
                        "image": image_key,
                        a: a_score,
                        b: b_score,
                    })

            pairwise[f"{a}_vs_{b}"] = {
                f"{a}_win_count": len(a_win),
                f"{b}_win_count": len(b_win),
                "tie_count": len(tie),
                f"{a}_win_images": a_win,
                f"{b}_win_images": b_win,
            }

    return pairwise


def write_summary_csv(results, output_path):
    rows = []

    for model_name, result in results.items():
        s = result["summary"]

        row = {
            "model": model_name,
            "total_images": s["basic"]["total_images"],
            "total_gt": s["basic"]["total_gt"],
            "total_pred": s["basic"]["total_pred"],
            "inference_error_images": s["basic"]["inference_error_images"],
            "eval_error_images": s["basic"]["eval_error_images"],
            "bbox_precision": s["bbox_match"]["precision"],
            "bbox_recall": s["bbox_match"]["recall"],
            "bbox_f1": s["bbox_match"]["f1"],
            "avg_iou": s["bbox_match"]["avg_iou"],
            "text_accuracy_on_matched": s["text_on_matched_bbox"]["accuracy_on_matched"],
            "end_to_end_precision": s["end_to_end"]["precision"],
            "end_to_end_recall": s["end_to_end"]["recall"],
            "end_to_end_f1": s["end_to_end"]["f1"],
            "漏检": s["error_type_counts"].get("漏检", 0),
            "无中生有": s["error_type_counts"].get("无中生有", 0),
            "字符错检": s["error_type_counts"].get("字符错检", 0),
            "多余字符": s["error_type_counts"].get("多余字符", 0),
            "内容缺失": s["error_type_counts"].get("内容缺失", 0),
            "total_errors": s["total_errors"],
        }

        rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(results, pairwise, output_path):
    lines = []

    lines.append("# 模型预测结果对比报告")
    lines.append("")
    lines.append(f"IoU 阈值：`{IOU_THRESHOLD}`")
    lines.append("")

    lines.append("## 1. 总体指标对比")
    lines.append("")
    lines.append(
        "| 模型 | 总GT | 总预测 | BBox F1 | Text Acc | End2End P | End2End R | End2End F1 | 漏检 | 无中生有 | 字符错检 |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for model_name, result in results.items():
        s = result["summary"]
        lines.append(
            "| {model} | {gt} | {pred} | {bbox_f1:.4f} | {text_acc:.4f} | {e2e_p:.4f} | {e2e_r:.4f} | {e2e_f1:.4f} | {miss} | {hall} | {char} |".format(
                model=model_name,
                gt=s["basic"]["total_gt"],
                pred=s["basic"]["total_pred"],
                bbox_f1=s["bbox_match"]["f1"],
                text_acc=s["text_on_matched_bbox"]["accuracy_on_matched"],
                e2e_p=s["end_to_end"]["precision"],
                e2e_r=s["end_to_end"]["recall"],
                e2e_f1=s["end_to_end"]["f1"],
                miss=s["error_type_counts"].get("漏检", 0),
                hall=s["error_type_counts"].get("无中生有", 0),
                char=s["error_type_counts"].get("字符错检", 0),
            )
        )

    lines.append("")
    lines.append("## 2. 逐图胜负对比")
    lines.append("")

    for pair_name, item in pairwise.items():
        lines.append(f"### {pair_name}")
        lines.append("")

        for k, v in item.items():
            if k.endswith("_count"):
                lines.append(f"- {k}: {v}")

        lines.append("")

    lines.append("## 3. 结论怎么看")
    lines.append("")
    lines.append("- `End2End F1` 最重要，表示 bbox 和文本同时正确的综合表现。")
    lines.append("- `BBox F1` 高但 `End2End F1` 低，说明定位可以，但文字识别错误多。")
    lines.append("- `无中生有` 多，说明模型乱输出。")
    lines.append("- `漏检` 多，说明模型保守或者小目标没识别出来。")
    lines.append("- DPO 常见现象是 precision 上升、recall 下降，所以要重点看 End2End F1 是否真的提升。")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_model_files(file_args):
    """
    解析：
    --files dpo=xxx.json sft=xxx.json doubao=xxx.json
    """

    pairs = []

    for x in file_args:
        if "=" not in x:
            raise ValueError(f"文件参数格式错误：{x}，应该是 name=path")

        name, path = x.split("=", 1)
        name = name.strip()
        path = path.strip()

        if not name or not path:
            raise ValueError(f"文件参数格式错误：{x}")

        pairs.append((name, path))

    return pairs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="多个预测文件，例如 dpo=output/predictions/dpo.json doubao=predictions_doubao.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/compare_models",
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    global IOU_THRESHOLD
    IOU_THRESHOLD = args.iou_threshold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_files = parse_model_files(args.files)

    results = {}

    for model_name, file_path in model_files:
        print("=" * 80)
        print(f"评估模型: {model_name}")
        print(f"文件路径: {file_path}")

        result = evaluate_file(
            model_name=model_name,
            file_path=file_path,
            iou_threshold=args.iou_threshold,
        )

        results[model_name] = result

        # 保存每个模型自己的错误文件
        error_output = output_dir / f"errors_{model_name}.json"
        with open(error_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": result["summary"],
                    "errors_by_image": result["errors_by_image"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"保存错误文件: {error_output}")

    pairwise = compare_pairwise(results)

    # 保存总 summary json
    summary_json = {
        model_name: result["summary"]
        for model_name, result in results.items()
    }

    with open(output_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    # 保存 csv
    write_summary_csv(results, output_dir / "model_summary.csv")

    # 保存 pairwise
    with open(output_dir / "pairwise_image_comparison.json", "w", encoding="utf-8") as f:
        json.dump(pairwise, f, ensure_ascii=False, indent=2)

    # 保存 markdown 报告
    write_markdown_report(results, pairwise, output_dir / "compare_report.md")

    print("=" * 80)
    print("全部对比完成")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    print("主要查看：")
    print(f"1. {output_dir / 'compare_report.md'}")
    print(f"2. {output_dir / 'model_summary.csv'}")
    print(f"3. {output_dir / 'pairwise_image_comparison.json'}")
    print(f"4. {output_dir / 'errors_模型名.json'}")


if __name__ == "__main__":
    main()