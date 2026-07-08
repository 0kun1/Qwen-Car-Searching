# Smart Car Searching：车位号识别与定位项目

本项目基于 **Qwen3-VL-2B-Instruct** 进行车位号识别与定位微调，目标是让多模态大模型从停车场图片中输出车位号文本及其二维位置框。

模型输入是一张图片，模型输出结构化 JSON：

```json
[
  {
    "text_content": "168",
    "bbox_2d": [833, 828, 962, 884]
  }
]
```

其中：

- `text_content`：识别出的车位号文本；
- `bbox_2d`：车位号在 1000×1000 图像坐标系下的位置框，格式为 `[x1, y1, x2, y2]`。

项目主流程包括：

```text
原始标注数据
→ SFT 数据构造
→ Qwen3-VL LoRA SFT 微调
→ SFT 模型预测
→ 预测结果评估与错误分析
→ DPO 偏好数据构造
→ DPO 偏好优化
→ DPO 模型预测与评估
→ 与 Doubao / Gemini / Qwen 等外部模型对比
```

---

## 1. 项目特点

- 使用 **Qwen3-VL-2B-Instruct** 作为视觉语言基础模型；
- 使用 **LLaMA-Factory** 进行 SFT 和 DPO 训练；
- 使用 **LoRA** 降低微调显存开销；
- 支持普通 SFT、增强 SFT、DPO 后训练对比；
- 支持 bbox 定位指标、文本识别指标、端到端指标；
- 支持逐图 badcase 分析；
- 支持与外部商业模型预测结果统一评估。

---

## 2. 目录结构

```text
car_search/
├── data/
│   ├── dataset_info.json
│   ├── train_labels_with_bbox.json
│   ├── test_labels_with_bbox.json
│   ├── train_sft.json
│   ├── test_sft.json
│   ├── train_sft_aug_correct.json
│   ├── train_sft_correct.json
│   ├── train_sft_cut_correct.json
│   ├── train_sft_perspective_correct.json
│   └── train_dpo.json
│
├── sft/
│   ├── prepare_sft_dataset.py
│   ├── qwen3vl_lora_sft_train_4090.yaml
│   ├── qwen3vl_lora_sft_train_aug.yaml
│   ├── qwen3vl_lora_sft_merge_4090.yaml
│   ├── qwen3vl_lora_sft_merge_aug.yaml
│   ├── train_sft.sh
│   └── train_sft_aug.sh
│
├── dpo/
│   ├── build_dpo_data.py
│   ├── build_train_dpo.py
│   ├── qwen3vl_lora_dpo_train.yaml
│   └── qwen3vl_lora_dpo_merge.yaml
│
├── augument/
│   ├── apply_crop_transform.py
│   └── apply_perspective_transform.py
│
├── analyze/
│   └── analyze_errors.py
│
├── scripts/
│   └── generate_dpo_and_eval_from_predictions.py
│
├── output/
│   ├── predictions/
│   ├── analysis/
│   └── compare_models/
│
├── predict.py
├── compare_predictions.py
├── draw.py
├── draw_badcases.py
├── register_aug_dataset.py
└── vllm_server.sh
```

说明：

- `data/`：存放原始标签、SFT 数据、DPO 数据；
- `sft/`：SFT 数据转换、训练和模型合并配置；
- `dpo/`：DPO 数据构造、DPO 训练和模型合并配置；
- `augument/`：数据增强脚本，包括裁剪增强和透视变换增强；
- `scripts/`：预测结果评估、DPO 候选构造等通用脚本；
- `output/predictions/`：模型预测结果；
- `output/analysis/`：评估报告和错误分析结果；
- `output/compare_models/`：多个模型之间的统一对比报告。

---

## 3. 环境依赖

推荐环境：

```text
Python 3.11
PyTorch
Transformers
LLaMA-Factory
qwen-vl-utils
Pillow
accelerate
peft
wandb 可选
vllm 可选
```

如果使用 LLaMA-Factory 训练，核心命令形式为：

```bash
llamafactory-cli train xxx.yaml
llamafactory-cli export xxx.yaml
```

本项目的训练配置默认适配单张 RTX 4090 级别显卡。显存不足时可以调小：

```yaml
per_device_train_batch_size
image_max_pixels
cutoff_len
```

---

## 4. 数据格式

### 4.1 原始标注格式

原始数据样本大致如下：

```json
{
  "raw_img": "原始图片路径.jpg",
  "label": 1,
  "answer": [
    {
      "label": "168",
      "type": "3",
      "有无车辆": "有",
      "bbox": {
        "xmin": 833.29,
        "ymin": 827.71,
        "xmax": 961.95,
        "ymax": 883.62
      }
    }
  ],
  "original_size": {
    "width": 4096,
    "height": 2304
  },
  "target_size": {
    "width": 1000,
    "height": 1000
  },
  "new_img": "test_images/test_0000.jpg"
}
```

### 4.2 SFT 数据格式

SFT 数据采用 ShareGPT 格式：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[{\"text_content\": text, \"bbox_2d\": [x1, y1, x2, y2]}]"
    },
    {
      "from": "gpt",
      "value": "[{\"text_content\": \"168\", \"bbox_2d\": [833, 828, 962, 884]}]"
    }
  ],
  "images": [
    "train_images/xxx.jpg"
  ]
}
```

### 4.3 DPO 数据格式

DPO 数据同样使用 ShareGPT ranking 格式：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>你是一个专业的车位号识别助手。请观察图片，输出车位号的位置与编号。输出json格式：[...]"
    }
  ],
  "chosen": {
    "from": "gpt",
    "value": "正确答案 JSON"
  },
  "rejected": {
    "from": "gpt",
    "value": "模型错误预测 JSON"
  },
  "images": [
    "train_images/xxx.jpg"
  ]
}
```

其中：

```text
chosen = GT 标注答案
rejected = SFT 模型错误预测
```

---

## 5. 运行流程

### 5.1 构造 SFT 数据

```bash
python sft/prepare_sft_dataset.py --data_dir data
```

该脚本会读取：

```text
data/train_labels_with_bbox.json
data/test_labels_with_bbox.json
```

并生成：

```text
data/train_sft.json
data/test_sft.json
```

---

### 5.2 训练普通 SFT 模型

```bash
llamafactory-cli train sft/qwen3vl_lora_sft_train_4090.yaml
```

训练输出默认保存到：

```text
saves/qwen3-vl-sft-lora-4090
```

---

### 5.3 训练增强 SFT 模型

```bash
llamafactory-cli train sft/qwen3vl_lora_sft_train_aug.yaml
```

增强版 SFT 使用的数据集是：

```text
train_parking_aug_correct
```

对应文件：

```text
data/train_sft_aug_correct.json
```

训练输出默认保存到：

```text
saves/qwen3-vl-sft-lora-aug-4090
```

---

### 5.4 合并 SFT LoRA 模型

普通 SFT：

```bash
llamafactory-cli export sft/qwen3vl_lora_sft_merge_4090.yaml
```

增强 SFT：

```bash
llamafactory-cli export sft/qwen3vl_lora_sft_merge_aug.yaml
```

合并后的模型一般保存到：

```text
saves/qwen3-vl-sft-merged-4090
saves/qwen3-vl-sft-merged-aug-4090
```

---

### 5.5 使用 SFT 模型预测 test 集

增强 SFT 示例：

```bash
python predict.py \
  --model_path saves/qwen3-vl-sft-merged-aug-4090 \
  --data_dir data \
  --task custom \
  --input_file data/test_sft.json \
  --output_file output/predictions/sft_aug_test_predictions_transformers.json
```

普通 SFT 示例：

```bash
python predict.py \
  --model_path saves/qwen3-vl-sft-merged-4090 \
  --data_dir data \
  --task custom \
  --input_file data/test_sft.json \
  --output_file output/predictions/sft_test_predictions_transformers.json
```

---

### 5.6 生成训练集预测结果

DPO 数据需要从训练集预测错误中构造，因此先对训练集增强版本进行预测：

```bash
python predict.py \
  --model_path saves/qwen3-vl-sft-merged-aug-4090 \
  --data_dir data \
  --task all
```

默认会生成：

```text
output/predictions/sft_train_correct_enable_vit_predictions.json
output/predictions/sft_train_cut_correct_enable_vit_predictions.json
output/predictions/sft_train_perspective_correct_enable_vit_predictions.json
```

---

### 5.7 构造 DPO 数据

```bash
python dpo/build_train_dpo.py \
  --input_files \
  output/predictions/sft_train_correct_enable_vit_predictions.json \
  output/predictions/sft_train_cut_correct_enable_vit_predictions.json \
  output/predictions/sft_train_perspective_correct_enable_vit_predictions.json \
  --output_file data/train_dpo.json \
  --extra_sample_num 500 \
  --seed 42
```

输出：

```text
data/train_dpo.json
```

注意：不要用 test 集错误样本构造训练数据，否则会造成测试集泄漏。

---

### 5.8 DPO 训练

```bash
llamafactory-cli train dpo/qwen3vl_lora_dpo_train.yaml
```

DPO 配置核心字段：

```yaml
stage: dpo
finetuning_type: lora
pref_loss: sigmoid
pref_beta: 0.1
pref_ftx: 0.2
learning_rate: 5.0e-6
num_train_epochs: 1.0
```

训练输出默认保存到：

```text
saves/qwen3-vl-dpo-lora-aug-4090
```

---

### 5.9 合并 DPO LoRA 模型

```bash
llamafactory-cli export dpo/qwen3vl_lora_dpo_merge.yaml
```

合并后的 DPO 模型默认保存到：

```text
saves/qwen3-vl-dpo-merged-aug-4090
```

---

### 5.10 使用 DPO 模型预测 test 集

```bash
python predict.py \
  --model_path saves/qwen3-vl-dpo-merged-aug-4090 \
  --data_dir data \
  --task custom \
  --input_file data/test_sft.json \
  --output_file output/predictions/dpo_test_predictions_transformers.json
```

---

## 6. 评估预测结果

### 6.1 单模型评估

```bash
python scripts/generate_dpo_and_eval_from_predictions.py \
  --input_file output/predictions/dpo_test_predictions_transformers.json \
  --output_dir output/analysis/dpo_test \
  --dpo_output_name dpo_test_candidates_ignore.json
```

输出：

```text
output/analysis/dpo_test/
├── eval_report.md
├── eval_summary.json
├── prediction_errors.json
├── dpo_test_candidates_ignore.json
└── dpo_test_candidates_ignore_meta.json
```

其中：

- `eval_report.md`：可读性最强的评估报告；
- `eval_summary.json`：结构化指标；
- `prediction_errors.json`：逐图错误分析；
- `dpo_test_candidates_ignore.json`：由 test 预测生成的候选文件，仅用于分析，不应用于训练。

---

### 6.2 多模型统一对比

如果有多个模型预测文件，例如：

```text
output/predictions/sft_test_predictions_transformers.json
output/predictions/sft_aug_test_predictions_transformers.json
output/predictions/dpo_test_predictions_transformers.json
output/predictions/predictions_doubao.json
output/predictions/predictions_gemini.json
output/predictions/predictions_qwen.json
```

可以运行：

```bash
python compare_predictions.py \
  --files \
  sft=output/predictions/sft_test_predictions_transformers.json \
  sft_aug=output/predictions/sft_aug_test_predictions_transformers.json \
  dpo=output/predictions/dpo_test_predictions_transformers.json \
  doubao=output/predictions/predictions_doubao.json \
  gemini=output/predictions/predictions_gemini.json \
  qwen=output/predictions/predictions_qwen.json \
  --output_dir output/compare_models
```

输出：

```text
output/compare_models/
├── compare_report.md
├── model_summary.csv
├── model_summary.json
├── pairwise_image_comparison.json
├── errors_sft.json
├── errors_sft_aug.json
├── errors_dpo.json
├── errors_doubao.json
├── errors_gemini.json
└── errors_qwen.json
```

---

## 7. 指标说明

### 7.1 BBox Precision / Recall / F1

只看位置框是否匹配。若预测框与 GT 框的 IoU 大于阈值，例如 0.5，则认为 bbox 匹配成功。

```text
BBox Precision = matched_bbox / total_pred_bbox
BBox Recall    = matched_bbox / total_gt_bbox
BBox F1        = 2 * Precision * Recall / (Precision + Recall)
```

### 7.2 Text Accuracy on Matched BBox

只在 bbox 已匹配的样本上，判断文本是否正确。

```text
Text Acc = bbox 匹配后文本正确数 / bbox 匹配数
```

### 7.3 End-to-End Precision / Recall / F1

端到端指标要求：

```text
bbox 位置正确 + text_content 文本正确
```

二者同时正确才算一个真正的 TP。

```text
End2End Precision = TP / (TP + FP)
End2End Recall    = TP / (TP + FN)
End2End F1        = 2 * Precision * Recall / (Precision + Recall)
```

在本任务中，`End2End F1` 是最重要的综合指标。

---

## 8. 当前实验结果

IoU 阈值：`0.5`

| 模型 | 总GT | 总预测 | BBox F1 | Text Acc | End2End P | End2End R | End2End F1 | 漏检 | 无中生有 | 字符错检 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sft | 542 | 541 | 0.9344 | 0.8814 | 0.8244 | 0.8229 | 0.8236 | 36 | 35 | 33 |
| sft_aug | 550 | 528 | 0.9332 | 0.9066 | 0.8636 | 0.8291 | 0.8460 | 47 | 25 | 27 |
| dpo | 550 | 521 | 0.9356 | 0.9042 | 0.8695 | 0.8236 | 0.8459 | 49 | 20 | 27 |
| doubao | 547 | 648 | 0.7833 | 0.7991 | 0.5772 | 0.6837 | 0.6259 | 79 | 180 | 23 |
| gemini | 550 | 710 | 0.3381 | 0.7887 | 0.2366 | 0.3055 | 0.2667 | 337 | 497 | 8 |
| qwen | 550 | 676 | 0.7879 | 0.7971 | 0.5695 | 0.7000 | 0.6281 | 67 | 193 | 26 |

主要结论：

- `sft_aug` 明显优于普通 `sft`，端到端 F1 从 `0.8236` 提升到 `0.8460`；
- `dpo` 相比 `sft_aug` 提升非常有限，端到端 F1 基本持平：`0.8460 → 0.8459`；
- `dpo` 的 Precision 更高，但 Recall 更低，说明模型更保守；
- `dpo` 的无中生有数量从 `25` 降到 `20`，但漏检从 `47` 增加到 `49`；
- 当前任务中，DPO 主要改变了 precision / recall trade-off，而不是显著提升整体能力。

---

## 9. Badcase 类型

评估脚本会把错误分为几类：

```text
漏检：GT 有，但模型没输出
无中生有：GT 没有，但模型多输出
字符错检：bbox 匹配成功，但文本字符识别错
内容缺失：预测文本少了部分字符
多余字符：预测文本多了部分字符
```

常见错误模式：

- 小目标、边缘目标、远处目标容易漏检；
- bbox 定位正确但数字或字母识别错误；
- text 对了但 bbox 偏移较大，导致 IoU 不达标；
- 外部模型容易识别车牌、背景文字或区域标识，导致无中生有。

---

## 10. 注意事项

### 10.1 测试集不能用于训练

`output/analysis/dpo_test/` 中虽然会生成类似 DPO 候选文件，但这是由 test 预测结果生成的，只能用于错误分析，不能用于训练。

正确做法是：

```text
train predictions → 构造 train_dpo.json → DPO 训练

test predictions → 只做评估和错误分析
```

### 10.2 当前公开包可能不包含图片和模型

如果仓库是脱敏版本，可能已经排除了：

```text
data/train_images/
data/test_images/
saves/
pretrained/
wandb/
模型权重文件
```

这种情况下，不能直接复现完整训练和预测。需要自行补充：

```text
1. 原始图片数据
2. 预训练 Qwen3-VL 模型
3. 已训练 LoRA 或重新训练
```

### 10.3 分享仓库前建议排除隐私文件

公开分享或上传 GitHub 前，建议排除：

```text
wandb/
.ipynb_checkpoints/
.git/
saves/
pretrained/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.jpg
*.png
*.webp
```

可以用下面命令检查压缩包是否仍包含敏感或大文件：

```bash
tar -tzf car_search_clean_no_private.tar.gz | grep -Ei 'wandb|\.git|ipynb_checkpoints|safetensors|\.bin|\.pt|\.pth|\.ckpt|\.jpg|\.png|\.webp'
```

如果没有输出，说明这些大类基本排除干净。

---

## 11. 后续改进方向

当前 DPO 效果有限，后续更值得尝试：

1. 构造漏检型 DPO 样本：`chosen = 完整 GT`，`rejected = 漏检预测`；
2. 构造 bbox 偏移 hard negative：文本正确但 IoU 不达标；
3. 对小目标、边缘目标和远景目标做更强数据增强；
4. 提高 `image_max_pixels`，改善小字识别；
5. 对多目标样本加权，减少只输出最大目标的问题；
6. 使用更严格的输出约束或后处理，减少无中生有和坏 JSON。

---

## 12. 常用命令汇总

```bash
# 1. 构造 SFT 数据
python sft/prepare_sft_dataset.py --data_dir data

# 2. 训练增强 SFT
llamafactory-cli train sft/qwen3vl_lora_sft_train_aug.yaml

# 3. 合并增强 SFT
llamafactory-cli export sft/qwen3vl_lora_sft_merge_aug.yaml

# 4. SFT test 预测
python predict.py \
  --model_path saves/qwen3-vl-sft-merged-aug-4090 \
  --data_dir data \
  --task custom \
  --input_file data/test_sft.json \
  --output_file output/predictions/sft_aug_test_predictions_transformers.json

# 5. 生成 train predictions，用于构造 DPO
python predict.py \
  --model_path saves/qwen3-vl-sft-merged-aug-4090 \
  --data_dir data \
  --task all

# 6. 构造 DPO 数据
python dpo/build_train_dpo.py \
  --input_files \
  output/predictions/sft_train_correct_enable_vit_predictions.json \
  output/predictions/sft_train_cut_correct_enable_vit_predictions.json \
  output/predictions/sft_train_perspective_correct_enable_vit_predictions.json \
  --output_file data/train_dpo.json \
  --extra_sample_num 500 \
  --seed 42

# 7. DPO 训练
llamafactory-cli train dpo/qwen3vl_lora_dpo_train.yaml

# 8. 合并 DPO
llamafactory-cli export dpo/qwen3vl_lora_dpo_merge.yaml

# 9. DPO test 预测
python predict.py \
  --model_path saves/qwen3-vl-dpo-merged-aug-4090 \
  --data_dir data \
  --task custom \
  --input_file data/test_sft.json \
  --output_file output/predictions/dpo_test_predictions_transformers.json

# 10. 单模型评估
python scripts/generate_dpo_and_eval_from_predictions.py \
  --input_file output/predictions/dpo_test_predictions_transformers.json \
  --output_dir output/analysis/dpo_test \
  --dpo_output_name dpo_test_candidates_ignore.json

# 11. 多模型对比
python compare_predictions.py \
  --files \
  sft=output/predictions/sft_test_predictions_transformers.json \
  sft_aug=output/predictions/sft_aug_test_predictions_transformers.json \
  dpo=output/predictions/dpo_test_predictions_transformers.json \
  doubao=output/predictions/predictions_doubao.json \
  gemini=output/predictions/predictions_gemini.json \
  qwen=output/predictions/predictions_qwen.json \
  --output_dir output/compare_models
```
