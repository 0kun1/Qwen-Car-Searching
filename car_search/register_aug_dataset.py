import json
from pathlib import Path


DATASET_INFO_PATH = Path("data/dataset_info.json")

BASE_DATASET_NAME = "train_parking_correct"
NEW_DATASET_NAME = "train_parking_aug_correct"
NEW_FILE_NAME = "train_sft_aug_correct.json"


def main():
    if not DATASET_INFO_PATH.exists():
        raise FileNotFoundError("找不到 data/dataset_info.json")

    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    if BASE_DATASET_NAME not in dataset_info:
        raise KeyError(
            f"dataset_info.json 里找不到 {BASE_DATASET_NAME}，"
            f"请先检查已有数据集名字。"
        )

    new_item = dict(dataset_info[BASE_DATASET_NAME])
    new_item["file_name"] = NEW_FILE_NAME

    dataset_info[NEW_DATASET_NAME] = new_item

    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"已注册新数据集: {NEW_DATASET_NAME}")
    print(f"对应文件: {NEW_FILE_NAME}")


if __name__ == "__main__":
    main()