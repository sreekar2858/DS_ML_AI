import os
import argparse
import kagglehub

import yaml
# Load dataset information from YAML file
with open(os.path.join(os.path.dirname(__file__), 'dataset_info.yaml'), 'r') as file:
    DATASETS = yaml.safe_load(file)

def download_datasets(level="beginner", download_path="./data/raw/"):
    os.makedirs(os.path.join(download_path, level), exist_ok=True)
    if level not in DATASETS:
        raise ValueError(
            f"Level {level} not recognized. Choose from {list(DATASETS.keys())}"
        )
    downloaded_paths = {}
    for dataset in DATASETS[level]:
        try:
            print(f"Downloading dataset {dataset} ...")
            cache_path = kagglehub.dataset_download(dataset, path=download_path)
            downloaded_paths[dataset] = cache_path
            print(f"Downloaded and extracted to: {cache_path}")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")
    return downloaded_paths


def main():
    parser = argparse.ArgumentParser(
        description="Batch download Kaggle datasets by difficulty level using kagglehub."
    )
    parser.add_argument(
        "--level",
        type=str,
        default="beginner",
        choices=DATASETS.keys(),
        help="Difficulty level of datasets to download: beginner, intermediate, advanced, multimodal",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/raw/",
        help="Directory to download datasets into (default: ./data/raw/)",
    )
    args = parser.parse_args()

    print(f"Downloading '{args.level}' level datasets into '{args.path}'...")
    download_datasets(level=args.level, download_path=args.path)


if __name__ == "__main__":
    main()
