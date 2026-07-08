import json
from pathlib import Path


INPUT_FILES = [
    "data/train_sft_correct.json",
    "data/train_sft_cut_correct.json",
    "data/train_sft_perspective_correct.json",
]

OUTPUT_FILE = "data/train_sft_aug_correct.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    all_data = []

    for path in INPUT_FILES:
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"找不到文件: {path}")

        data = load_json(path)

        print(f"{path}: {len(data)} 条")

        all_data.extend(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"合并完成，总样本数: {len(all_data)}")
    print(f"保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()