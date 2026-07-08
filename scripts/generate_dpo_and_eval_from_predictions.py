# -*- coding: utf-8 -*-

import argparse
import json
import difflib
from pathlib import Path
from collections import defaultdict, Counter


DEFAULT_PROMPT = (
    "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。"
    "输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}, "
    "{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"
)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def calculate_iou(bbox1, bbox2):
    """
    计算两个 bbox 的 IoU。
    bbox 格式：[x1, y1, x2, y2]
    """

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

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def normalize_items(items):
    """
    统一清洗 gt / pred 格式。
    保证每个元素是：
    {
      "text_content": str,
      "bbox_2d": [x1, y1, x2, y2]
    }
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


def match_predictions(gt_list, pred_list, iou_threshold=0.5):
    """
    和你之前 analyze_errors.py 的逻辑一致：
    1. 先按 bbox IoU 匹配 gt 和 pred
    2. 一个 gt 只能匹配一个 pred
    3. 一个 pred 只能匹配一个 gt
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
                # 负号是为了排序时 IoU 大的排前面
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
    细分文本错误类型：
    - 字符错检
    - 内容缺失
    - 多余字符
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


def analyze_one_item(item, iou_threshold=0.5):
    """
    分析单张图片：
    返回是否有错误，以及错误详情。
    """

    image_path = item["image_path"]
    image_name = image_path.split("/")[-1]

    gt_list = normalize_items(item.get("gt", []))
    pred_list = normalize_items(item.get("pred", []))
    infer_error = item.get("error", None)

    matched, unmatched_gts, unmatched_preds = match_predictions(
        gt_list=gt_list,
        pred_list=pred_list,
        iou_threshold=iou_threshold
    )

    errors = []
    correct_text_count = 0
    wrong_text_count = 0
    total_iou = 0.0

    for gt, pred, iou in matched:
        total_iou += iou

        gt_text = gt["text_content"]
        pred_text = pred["text_content"]

        if gt_text == pred_text:
            correct_text_count += 1
        else:
            wrong_text_count += 1
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

    for gt in unmatched_gts:
        errors.append({
            "type": "missing",
            "error_type": "漏检",
            "gt": gt["text_content"],
            "gt_bbox": gt["bbox_2d"],
        })

    for pred in unmatched_preds:
        errors.append({
            "type": "hallucination",
            "error_type": "无中生有",
            "pred": pred["text_content"],
            "pred_bbox": pred["bbox_2d"],
        })

    if infer_error is not None:
        errors.append({
            "type": "inference_error",
            "error_type": "推理报错",
            "details": str(infer_error),
        })

    result = {
        "image_path": image_path,
        "image_name": image_name,
        "gt": gt_list,
        "pred": pred_list,
        "matched": matched,
        "unmatched_gts": unmatched_gts,
        "unmatched_preds": unmatched_preds,
        "correct_text_count": correct_text_count,
        "wrong_text_count": wrong_text_count,
        "total_iou": total_iou,
        "errors": errors,
        "has_error": len(errors) > 0,
    }

    return result


def build_dpo_item(image_path, gt_list, pred_list, prompt=DEFAULT_PROMPT):
    """
    构造 LLaMAFactory DPO 所需 sharegpt ranking 格式。
    chosen = gt
    rejected = pred
    """

    return {
        "conversations": [
            {
                "from": "human",
                "value": prompt
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


def build_eval_summary(data, analyzed_items, iou_threshold=0.5):
    stats = {
        "total_images": 0,
        "total_gt": 0,
        "total_pred": 0,

        "matched": 0,
        "correct_text": 0,
        "wrong_text": 0,

        "total_iou": 0.0,

        "inference_error_images": 0,
        "eval_error_images": 0,

        "gt_empty_pred_empty": 0,
        "gt_nonempty_pred_empty": 0,
        "gt_empty_pred_nonempty": 0,
        "gt_nonempty_pred_nonempty": 0,

        "error_types": Counter(),
    }

    errors_by_image = {}

    for raw_item, analyzed in zip(data, analyzed_items):
        gt_list = analyzed["gt"]
        pred_list = analyzed["pred"]

        stats["total_images"] += 1
        stats["total_gt"] += len(gt_list)
        stats["total_pred"] += len(pred_list)

        if raw_item.get("error", None) is not None:
            stats["inference_error_images"] += 1

        if len(gt_list) == 0 and len(pred_list) == 0:
            stats["gt_empty_pred_empty"] += 1
        elif len(gt_list) > 0 and len(pred_list) == 0:
            stats["gt_nonempty_pred_empty"] += 1
        elif len(gt_list) == 0 and len(pred_list) > 0:
            stats["gt_empty_pred_nonempty"] += 1
        else:
            stats["gt_nonempty_pred_nonempty"] += 1

        stats["matched"] += len(analyzed["matched"])
        stats["correct_text"] += analyzed["correct_text_count"]
        stats["wrong_text"] += analyzed["wrong_text_count"]
        stats["total_iou"] += analyzed["total_iou"]

        if analyzed["has_error"]:
            stats["eval_error_images"] += 1
            errors_by_image[analyzed["image_name"]] = analyzed["errors"]

            for err in analyzed["errors"]:
                stats["error_types"][err["error_type"]] += 1

    total_gt = stats["total_gt"]
    total_pred = stats["total_pred"]
    matched = stats["matched"]
    correct_text = stats["correct_text"]

    bbox_precision = safe_div(matched, total_pred)
    bbox_recall = safe_div(matched, total_gt)
    bbox_f1 = safe_div(
        2 * bbox_precision * bbox_recall,
        bbox_precision + bbox_recall
    )

    text_acc_on_matched = safe_div(correct_text, matched)

    e2e_precision = safe_div(correct_text, total_pred)
    e2e_recall = safe_div(correct_text, total_gt)
    e2e_f1 = safe_div(
        2 * e2e_precision * e2e_recall,
        e2e_precision + e2e_recall
    )

    avg_iou = safe_div(stats["total_iou"], matched)

    summary = {
        "iou_threshold": iou_threshold,

        "basic": {
            "total_images": stats["total_images"],
            "total_gt": total_gt,
            "total_pred": total_pred,
            "inference_error_images": stats["inference_error_images"],
            "eval_error_images": stats["eval_error_images"],
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
        "total_errors": sum(stats["error_types"].values()),
    }

    return summary, errors_by_image


def build_markdown_report(input_file, summary):
    bbox = summary["bbox_match"]
    text = summary["text_on_matched_bbox"]
    e2e = summary["end_to_end"]
    basic = summary["basic"]
    empty = summary["empty_stats"]

    lines = []

    lines.append("# Test 效果总结报告")
    lines.append("")
    lines.append(f"输入文件：`{input_file}`")
    lines.append(f"IoU 阈值：`{summary['iou_threshold']}`")
    lines.append("")

    lines.append("## 1. 基本统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 总图片数 | {basic['total_images']} |")
    lines.append(f"| 总 GT 数 | {basic['total_gt']} |")
    lines.append(f"| 总预测数 | {basic['total_pred']} |")
    lines.append(f"| 推理报错图片数 | {basic['inference_error_images']} |")
    lines.append(f"| 评估发现有错误的图片数 | {basic['eval_error_images']} |")
    lines.append("")

    lines.append("## 2. 空值统计")
    lines.append("")
    lines.append("| 类型 | 数量 |")
    lines.append("|---|---:|")
    lines.append(f"| gt 空，pred 空 | {empty['gt_empty_pred_empty']} |")
    lines.append(f"| gt 非空，pred 空 | {empty['gt_nonempty_pred_empty']} |")
    lines.append(f"| gt 空，pred 非空 | {empty['gt_empty_pred_nonempty']} |")
    lines.append(f"| gt 非空，pred 非空 | {empty['gt_nonempty_pred_nonempty']} |")
    lines.append("")

    lines.append("## 3. BBox 匹配指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| matched | {bbox['matched']} |")
    lines.append(f"| precision | {bbox['precision']:.4f} |")
    lines.append(f"| recall | {bbox['recall']:.4f} |")
    lines.append(f"| f1 | {bbox['f1']:.4f} |")
    lines.append(f"| avg_iou | {bbox['avg_iou']:.4f} |")
    lines.append("")

    lines.append("## 4. 文本识别指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| bbox 匹配后文本正确数 | {text['correct_text']} |")
    lines.append(f"| bbox 匹配后文本错误数 | {text['wrong_text']} |")
    lines.append(f"| bbox 匹配后文本准确率 | {text['accuracy_on_matched']:.4f} |")
    lines.append("")

    lines.append("## 5. 端到端指标 bbox + text")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| TP | {e2e['tp']} |")
    lines.append(f"| FP | {e2e['fp']} |")
    lines.append(f"| FN | {e2e['fn']} |")
    lines.append(f"| precision | {e2e['precision']:.4f} |")
    lines.append(f"| recall | {e2e['recall']:.4f} |")
    lines.append(f"| f1 | {e2e['f1']:.4f} |")
    lines.append("")

    lines.append("## 6. 错误类型统计")
    lines.append("")
    lines.append("| 错误类型 | 数量 |")
    lines.append("|---|---:|")

    for error_type, count in sorted(
        summary["error_type_counts"].items(),
        key=lambda x: -x[1]
    ):
        lines.append(f"| {error_type} | {count} |")

    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入预测文件，例如 output/predictions/sft_aug_test_predictions_transformers.json"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/generated_from_predictions",
        help="输出目录"
    )

    parser.add_argument(
        "--dpo_output_name",
        type=str,
        default="dpo_candidates_from_predictions.json",
        help="DPO 输出文件名"
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="bbox 匹配 IoU 阈值"
    )

    parser.add_argument(
        "--dpo_mode",
        type=str,
        default="error_only",
        choices=["error_only", "all"],
        help="error_only 只把错误样本做成 DPO；all 会尝试全部样本"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="DPO conversations 里的 human prompt"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    analyzed_items = []
    dpo_dataset = []
    dpo_dataset_with_meta = []

    for item in data:
        analyzed = analyze_one_item(
            item=item,
            iou_threshold=args.iou_threshold
        )
        analyzed_items.append(analyzed)

        gt_list = analyzed["gt"]
        pred_list = analyzed["pred"]
        image_path = analyzed["image_path"]

        # DPO 不建议加入 chosen 和 rejected 完全相同的样本
        if gt_list == pred_list:
            continue

        if args.dpo_mode == "error_only":
            if not analyzed["has_error"]:
                continue

        dpo_item = build_dpo_item(
            image_path=image_path,
            gt_list=gt_list,
            pred_list=pred_list,
            prompt=args.prompt
        )

        dpo_dataset.append(dpo_item)

        dpo_dataset_with_meta.append({
            "index": item.get("index", None),
            "image_path": image_path,
            "errors": analyzed["errors"],
            "dpo": dpo_item,
        })

    summary, errors_by_image = build_eval_summary(
        data=data,
        analyzed_items=analyzed_items,
        iou_threshold=args.iou_threshold
    )

    # 1. 保存 DPO 数据
    dpo_output_path = output_dir / args.dpo_output_name
    with open(dpo_output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=2)

    # 2. 保存带错误信息的 DPO meta 文件，方便你检查
    dpo_meta_output_path = output_dir / args.dpo_output_name.replace(".json", "_meta.json")
    with open(dpo_meta_output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset_with_meta, f, ensure_ascii=False, indent=2)

    # 3. 保存评估 summary
    summary_output_path = output_dir / "eval_summary.json"
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 4. 保存详细错误报告
    errors_output_path = output_dir / "prediction_errors.json"
    with open(errors_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "errors_by_image": errors_by_image
        }, f, ensure_ascii=False, indent=2)

    # 5. 保存 markdown 报告
    report_md = build_markdown_report(
        input_file=str(input_path),
        summary=summary
    )

    report_output_path = output_dir / "eval_report.md"
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("=" * 80)
    print("生成完成")
    print("=" * 80)
    print(f"输入文件: {input_path}")
    print(f"DPO 数据: {dpo_output_path}")
    print(f"DPO meta 数据: {dpo_meta_output_path}")
    print(f"评估 summary: {summary_output_path}")
    print(f"详细错误报告: {errors_output_path}")
    print(f"Markdown 报告: {report_output_path}")
    print()

    print("-" * 80)
    print("核心指标")
    print("-" * 80)
    print(f"总图片数: {summary['basic']['total_images']}")
    print(f"总 GT 数: {summary['basic']['total_gt']}")
    print(f"总预测数: {summary['basic']['total_pred']}")
    print(f"DPO 样本数: {len(dpo_dataset)}")
    print(f"评估错误图片数: {summary['basic']['eval_error_images']}")
    print(f"BBox F1: {summary['bbox_match']['f1']:.4f}")
    print(f"文本准确率: {summary['text_on_matched_bbox']['accuracy_on_matched']:.4f}")
    print(f"端到端 F1: {summary['end_to_end']['f1']:.4f}")


if __name__ == "__main__":
    main()