import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a smaller sampled deepfake dataset from the Kaggle extracted frames layout."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Source dataset root containing train, val, and test folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_small"),
        help="Destination directory for the sampled dataset.",
    )
    parser.add_argument(
        "--train-real",
        type=int,
        default=1000,
        help="Number of real images to copy into the train split.",
    )
    parser.add_argument(
        "--train-fake",
        type=int,
        default=1000,
        help="Number of fake images to copy into the train split.",
    )
    parser.add_argument(
        "--val-real",
        type=int,
        default=250,
        help="Number of real images to copy into the val split.",
    )
    parser.add_argument(
        "--val-fake",
        type=int,
        default=250,
        help="Number of fake images to copy into the val split.",
    )
    parser.add_argument(
        "--test-real",
        type=int,
        default=250,
        help="Number of real images to copy into the test split.",
    )
    parser.add_argument(
        "--test-fake",
        type=int,
        default=250,
        help="Number of fake images to copy into the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Copy images directly into split/real and split/fake folders instead of preserving source subfolders.",
    )
    return parser.parse_args()


def split_targets(args):
    return {
        "train": {"real": args.train_real, "fake": args.train_fake},
        "val": {"real": args.val_real, "fake": args.val_fake},
        "test": {"real": args.test_real, "fake": args.test_fake},
    }


def classify_folder(folder_name: str):
    lowered = folder_name.lower()
    if lowered.startswith("real"):
        return "real"
    if lowered.startswith("fake"):
        return "fake"
    return None


def gather_images(split_dir: Path):
    buckets = {"real": [], "fake": []}
    for folder in split_dir.iterdir():
        if not folder.is_dir():
            continue
        label = classify_folder(folder.name)
        if label is None:
            continue
        for image_path in folder.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                buckets[label].append(image_path)
    return buckets


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_sampled_images(image_paths, destination_dir: Path, flatten: bool):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_paths):
        if flatten:
            target_name = f"{index:05d}_{image_path.parent.name}_{image_path.name}"
            shutil.copy2(image_path, destination_dir / target_name)
        else:
            target_subdir = destination_dir / image_path.parent.name
            target_subdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_subdir / image_path.name)


def main():
    args = parse_args()
    random.seed(args.seed)

    targets = split_targets(args)
    ensure_clean_dir(args.output_dir)

    for split, class_targets in targets.items():
        split_dir = args.input_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        split_buckets = gather_images(split_dir)
        print(f"\nPreparing split: {split}")

        for label, requested_count in class_targets.items():
            available = split_buckets[label]
            if not available:
                raise ValueError(f"No {label} images found in {split_dir}")

            sample_size = min(requested_count, len(available))
            sampled = random.sample(available, sample_size)
            destination = args.output_dir / split / label
            copy_sampled_images(sampled, destination, flatten=args.flatten)
            print(f"  {label}: copied {sample_size} images to {destination}")

    print(f"\nSmall dataset created at: {args.output_dir}")


if __name__ == "__main__":
    main()
