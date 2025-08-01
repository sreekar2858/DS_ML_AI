import os
import argparse
import kagglehub

# Map of dataset levels to Kaggle dataset slugs (update these slugs as needed!)
DATASETS = {
    "beginner": [
        "uciml/iris",
        "abhijithudayakumar/the-boston-housing-dataset",
        "paultimothymooney/chest-xray-pneumonia",  # Substitute for MNIST if no MNIST Kaggle slug
        "heptapod/titanic",
        "uciml/wine-quality",
        "shubhendra7/diamonds",
    ],
    "intermediate": [
        "tensorflow-datasets/cifar10",
        "zalando-research/fashionmnist",
        "catherineh/dog-breed-identification",
        "uciml/pima-indians-diabetes-database",
        "ashishpatel26/sentiment-analysis-dataset",
        "josesalasbank/bank-marketing-dataset",
    ],
    "advanced": [
        "imagenet-object-localization-challenge",
        "coco-dataset/coco-2017-dataset",
        "cityscapes/cityscapes-dataset",
        "kayichan/vqa-dataset",
        "alxmamaev/open-images-dataset",
        "mozillaorg/common-voice",
        "shubhendra7/segment-anything-dataset",
    ],
    "multimodal": [
        "kayichan/vqa-dataset",
        "asbdr67/egotv-dataset",
        "worldbank/world-development-indicators",
    ],
}

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
            path = kagglehub.dataset_download(dataset, path=download_path, unzip=True)
            downloaded_paths[dataset] = path
            print(f"Downloaded and extracted to: {path}")
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
