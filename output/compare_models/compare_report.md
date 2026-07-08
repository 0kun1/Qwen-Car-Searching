# 模型预测结果对比报告

IoU 阈值：`0.5`

## 1. 总体指标对比

| 模型 | 总GT | 总预测 | BBox F1 | Text Acc | End2End P | End2End R | End2End F1 | 漏检 | 无中生有 | 字符错检 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sft | 542 | 541 | 0.9344 | 0.8814 | 0.8244 | 0.8229 | 0.8236 | 36 | 35 | 33 |
| sft_aug | 550 | 528 | 0.9332 | 0.9066 | 0.8636 | 0.8291 | 0.8460 | 47 | 25 | 27 |
| dpo | 550 | 521 | 0.9356 | 0.9042 | 0.8695 | 0.8236 | 0.8459 | 49 | 20 | 27 |
| doubao | 547 | 648 | 0.7833 | 0.7991 | 0.5772 | 0.6837 | 0.6259 | 79 | 180 | 23 |
| gemini | 550 | 710 | 0.3381 | 0.7887 | 0.2366 | 0.3055 | 0.2667 | 337 | 497 | 8 |
| qwen | 550 | 676 | 0.7879 | 0.7971 | 0.5695 | 0.7000 | 0.6281 | 67 | 193 | 26 |

## 2. 逐图胜负对比

### sft_vs_sft_aug

- sft_win_count: 30
- sft_aug_win_count: 39
- tie_count: 431

### sft_vs_dpo

- sft_win_count: 32
- dpo_win_count: 40
- tie_count: 428

### sft_vs_doubao

- sft_win_count: 153
- doubao_win_count: 40
- tie_count: 307

### sft_vs_gemini

- sft_win_count: 257
- gemini_win_count: 18
- tie_count: 225

### sft_vs_qwen

- sft_win_count: 153
- qwen_win_count: 38
- tie_count: 309

### sft_aug_vs_dpo

- sft_aug_win_count: 4
- dpo_win_count: 4
- tie_count: 492

### sft_aug_vs_doubao

- sft_aug_win_count: 165
- doubao_win_count: 40
- tie_count: 295

### sft_aug_vs_gemini

- sft_aug_win_count: 273
- gemini_win_count: 15
- tie_count: 212

### sft_aug_vs_qwen

- sft_aug_win_count: 164
- qwen_win_count: 38
- tie_count: 298

### dpo_vs_doubao

- dpo_win_count: 169
- doubao_win_count: 42
- tie_count: 289

### dpo_vs_gemini

- dpo_win_count: 275
- gemini_win_count: 16
- tie_count: 209

### dpo_vs_qwen

- dpo_win_count: 167
- qwen_win_count: 40
- tie_count: 293

### doubao_vs_gemini

- doubao_win_count: 197
- gemini_win_count: 45
- tie_count: 258

### doubao_vs_qwen

- doubao_win_count: 27
- qwen_win_count: 34
- tie_count: 439

### gemini_vs_qwen

- gemini_win_count: 48
- qwen_win_count: 201
- tie_count: 251

## 3. 结论怎么看

- `End2End F1` 最重要，表示 bbox 和文本同时正确的综合表现。
- `BBox F1` 高但 `End2End F1` 低，说明定位可以，但文字识别错误多。
- `无中生有` 多，说明模型乱输出。
- `漏检` 多，说明模型保守或者小目标没识别出来。
- DPO 常见现象是 precision 上升、recall 下降，所以要重点看 End2End F1 是否真的提升。
