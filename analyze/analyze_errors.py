# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict, Counter
import difflib


input_file = "output/predictions/sft_aug_test_predictions_transformers.json"

output_file = "output/analysis/sft_aug_test_transformers_predictions_errors.json"
summary_file = "output/analysis/sft_aug_test_transformers_eval_summary.json"

IOU_THRESHOLD = 0.5


def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - inter_area

    if union_area > 0:
        return inter_area / union_area
    else:
        return 0.0


def match_predictions(gt_list, pred_list, iou_threshold=IOU_THRESHOLD):
    """
    沿用原项目逻辑：
    1. 先根据 bbox IoU 匹配 gt 和 pred
    2. 一个 gt 只能匹配一个 pred
    3. 一个 pred 也只能匹配一个 gt
    4. 匹配后再判断 text_content 是否正确
    """
    matched = []
    used_gt_indices = set()
    used_pred_indices = set()

    iou_pairs = []

    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            if "bbox_2d" not in gt or "bbox_2d" not in pred:
                continue

            iou = calculate_iou(gt["bbox_2d"], pred["bbox_2d"])

            if iou >= iou_threshold:
                iou_pairs.append((-iou, i, j, gt, pred))

    iou_pairs.sort()

    for neg_iou, i, j, gt, pred in iou_pairs:
        if i not in used_gt_indices and j not in used_pred_indices:
            matched.append((gt, pred, -neg_iou))
            used_gt_indices.add(i)
            used_pred_indices.add(j)

    unmatched_gts = [
        gt for i, gt in enumerate(gt_list)
        if i not in used_gt_indices
    ]

    unmatched_preds = [
        pred for j, pred in enumerate(pred_list)
        if j not in used_pred_indices
    ]

    return matched, unmatched_gts, unmatched_preds


def classify_text_error(gt_text, pred_text):
    """
    沿用原项目逻辑：
    对比 gt_text 和 pred_text，判断字符错检、内容缺失、多余字符。
    """
    if gt_text == pred_text:
        return "正确", ""

    diff = list(difflib.ndiff(gt_text, pred_text))

    missing_chars = []
    extra_chars = []
    replaced_chars = []

    i = 0

    while i < len(diff):
        if diff[i].startswith("- "):
            missing_char = diff[i][2]

            if i + 1 < len(diff) and diff[i + 1].startswith("+ "):
                replaced_chars.append((missing_char, diff[i + 1][2]))
                i += 2
            else:
                missing_chars.append(missing_char)
                i += 1

        elif diff[i].startswith("+ "):
            extra_chars.append(diff[i][2])
            i += 1

        else:
            i += 1

    error_types = []
    details = []

    if replaced_chars:
        replace_desc = ", ".join([f"{g}->{p}" for g, p in replaced_chars])
        error_types.append("字符错检")
        details.append(f"字符替换: {replace_desc}")

    if missing_chars:
        error_types.append("内容缺失")
        details.append(f"缺失字符: {''.join(missing_chars)}")

    if extra_chars:
        error_types.append("多余字符")
        details.append(f"多余字符: {''.join(extra_chars)}")

    if not error_types:
        error_types.append("其他错误")

    return "|".join(error_types), "; ".join(details)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def analyze_predictions(data):
    stats = {
        "total_images": 0,
        "total_gt": 0,
        "total_pred": 0,

        "matched": 0,
        "correct_text": 0,
        "wrong_text": 0,

        "total_iou": 0.0,
        "ious": [],

        "error_images": 0,

        "gt_empty_pred_empty": 0,
        "gt_nonempty_pred_empty": 0,
        "gt_empty_pred_nonempty": 0,
        "gt_nonempty_pred_nonempty": 0,

        "error_types": Counter(),
        "image_errors": defaultdict(list),
        "details_by_image": defaultdict(list),
    }

    for item in data:
        image_path = item["image_path"]
        image_name = image_path.split("/")[-1]

        gt_list = item.get("gt", [])
        pred_list = item.get("pred", [])
        error = item.get("error", None)

        stats["total_images"] += 1
        stats["total_gt"] += len(gt_list)
        stats["total_pred"] += len(pred_list)

        if error is not None:
            stats["error_images"] += 1

        if len(gt_list) == 0 and len(pred_list) == 0:
            stats["gt_empty_pred_empty"] += 1
        elif len(gt_list) > 0 and len(pred_list) == 0:
            stats["gt_nonempty_pred_empty"] += 1
        elif len(gt_list) == 0 and len(pred_list) > 0:
            stats["gt_empty_pred_nonempty"] += 1
        else:
            stats["gt_nonempty_pred_nonempty"] += 1

        matched, unmatched_gts, unmatched_preds = match_predictions(
            gt_list,
            pred_list,
            iou_threshold=IOU_THRESHOLD,
        )

        stats["matched"] += len(matched)

        for gt, pred, iou in matched:
            gt_text = gt["text_content"]
            pred_text = pred["text_content"]

            stats["ious"].append(iou)
            stats["total_iou"] += iou

            if gt_text == pred_text:
                stats["correct_text"] += 1
            else:
                stats["wrong_text"] += 1

                error_type, details = classify_text_error(gt_text, pred_text)
                stats["error_types"][error_type] += 1

                stats["image_errors"][image_name].append({
                    "type": "text_error",
                    "gt": gt_text,
                    "pred": pred_text,
                    "error_type": error_type,
                    "details": details,
                    "iou": iou,
                    "gt_bbox": gt["bbox_2d"],
                    "pred_bbox": pred["bbox_2d"],
                })

                stats["details_by_image"][image_name].append(
                    f"文本错误: GT='{gt_text}', PRED='{pred_text}' ({error_type}: {details}), IoU={iou:.3f}"
                )

        for gt in unmatched_gts:
            stats["error_types"]["漏检"] += 1

            stats["image_errors"][image_name].append({
                "type": "missing",
                "gt": gt["text_content"],
                "gt_bbox": gt["bbox_2d"],
            })

            stats["details_by_image"][image_name].append(
                f"漏检: GT='{gt['text_content']}' at {gt['bbox_2d']}"
            )

        for pred in unmatched_preds:
            stats["error_types"]["无中生有"] += 1

            stats["image_errors"][image_name].append({
                "type": "hallucination",
                "pred": pred["text_content"],
                "pred_bbox": pred["bbox_2d"],
            })

            stats["details_by_image"][image_name].append(
                f"无中生有: PRED='{pred['text_content']}' at {pred['bbox_2d']}"
            )

    return stats


def build_summary(stats):
    total_gt = stats["total_gt"]
    total_pred = stats["total_pred"]
    matched = stats["matched"]
    correct_text = stats["correct_text"]

    bbox_precision = safe_div(matched, total_pred)
    bbox_recall = safe_div(matched, total_gt)
    bbox_f1 = safe_div(2 * bbox_precision * bbox_recall, bbox_precision + bbox_recall)

    text_acc_on_matched = safe_div(correct_text, matched)

    # 端到端：既要 bbox 匹配，又要 text 正确
    e2e_precision = safe_div(correct_text, total_pred)
    e2e_recall = safe_div(correct_text, total_gt)
    e2e_f1 = safe_div(2 * e2e_precision * e2e_recall, e2e_precision + e2e_recall)

    avg_iou = safe_div(stats["total_iou"], matched)

    total_errors = sum(stats["error_types"].values())

    summary = {
        "input_file": input_file,
        "iou_threshold": IOU_THRESHOLD,

        "basic": {
            "total_images": stats["total_images"],
            "total_gt": total_gt,
            "total_pred": total_pred,
            "error_images": stats["error_images"],
        },

        "empty_stats": {
            "gt_empty_pred_empty": stats["gt_empty_pred_empty"],
            "gt_nonempty_pred_empty": stats["gt_nonempty_pred_empty"],
            "gt_empty_pred_nonempty": stats["gt_empty_pred_nonempty"],
            "gt_nonempty_pred_nonempty": stats["gt_nonempty_pred_nonempty"],
        },

        "bbox_match": {
            "matched": matched,
            "precision": bbox_precision,
            "recall": bbox_recall,
            "f1": bbox_f1,
            "avg_iou": avg_iou,
        },

        "text_on_matched_bbox": {
            "correct_text": correct_text,
            "wrong_text": stats["wrong_text"],
            "accuracy_on_matched": text_acc_on_matched,
        },

        "end_to_end": {
            "tp": correct_text,
            "fp": total_pred - correct_text,
            "fn": total_gt - correct_text,
            "precision": e2e_precision,
            "recall": e2e_recall,
            "f1": e2e_f1,
        },

        "error_type_counts": dict(stats["error_types"]),
        "total_errors": total_errors,
    }

    return summary


def print_summary(stats, summary):
    print("=" * 80)
    print("车位号识别正式评估报告")
    print("=" * 80)

    print(f"预测文件: {input_file}")
    print(f"IoU阈值: {IOU_THRESHOLD}")
    print()

    print("-" * 80)
    print("基本统计")
    print("-" * 80)
    print(f"总图片数: {stats['total_images']}")
    print(f"总GT标注数: {stats['total_gt']}")
    print(f"总预测数: {stats['total_pred']}")
    print(f"推理报错图片数: {stats['error_images']}")
    print()

    print("-" * 80)
    print("空值统计")
    print("-" * 80)
    print(f"gt为空 且 pred为空: {stats['gt_empty_pred_empty']}")
    print(f"gt不为空 但 pred为空，也就是漏检图片: {stats['gt_nonempty_pred_empty']}")
    print(f"gt为空 但 pred不为空，也就是误检图片: {stats['gt_empty_pred_nonempty']}")
    print(f"gt不为空 且 pred不为空: {stats['gt_nonempty_pred_nonempty']}")
    print()

    print("-" * 80)
    print("bbox 匹配指标")
    print("-" * 80)
    bbox = summary["bbox_match"]
    print(f"匹配成功数: {bbox['matched']}")
    print(f"Precision: {bbox['precision']:.4f}")
    print(f"Recall: {bbox['recall']:.4f}")
    print(f"F1: {bbox['f1']:.4f}")
    print(f"平均 IoU: {bbox['avg_iou']:.4f}")
    print()

    print("-" * 80)
    print("文本识别指标")
    print("-" * 80)
    text = summary["text_on_matched_bbox"]
    print(f"bbox匹配后，文本完全正确数: {text['correct_text']}")
    print(f"bbox匹配后，文本错误数: {text['wrong_text']}")
    print(f"bbox匹配后的文本准确率: {text['accuracy_on_matched']:.4f}")
    print()

    print("-" * 80)
    print("端到端指标 text + bbox")
    print("-" * 80)
    e2e = summary["end_to_end"]
    print(f"TP: {e2e['tp']}")
    print(f"FP: {e2e['fp']}")
    print(f"FN: {e2e['fn']}")
    print(f"Precision: {e2e['precision']:.4f}")
    print(f"Recall: {e2e['recall']:.4f}")
    print(f"F1: {e2e['f1']:.4f}")
    print()

    print("-" * 80)
    print("错误类型统计")
    print("-" * 80)
    total_errors = summary["total_errors"]

    for error_type, count in sorted(stats["error_types"].items(), key=lambda x: -x[1]):
        percentage = safe_div(count, total_errors) * 100
        print(f"{error_type:20s}: {count:5d} ({percentage:5.1f}%)")

    print(f"{'总计':20s}: {total_errors:5d}")
    print()

    print("-" * 80)
    print("前20张错误图片")
    print("-" * 80)

    error_images = sorted(
        stats["details_by_image"].items(),
        key=lambda x: -len(x[1])
    )[:20]

    for image_name, errors in error_images:
        print(f"\n{image_name}:")
        for error in errors:
            print(f"  - {error}")


def save_report(stats, summary):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    detailed_report = {
        "summary": summary,
        "errors_by_image": dict(stats["image_errors"]),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细错误报告已保存到: {output_file}")
    print(f"评估摘要已保存到: {summary_file}")


def main():
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = analyze_predictions(data)
    summary = build_summary(stats)

    print_summary(stats, summary)
    save_report(stats, summary)


if __name__ == "__main__":
    main()