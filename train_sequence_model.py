import argparse
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import tensorflow as tf


IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a video-sequence deepfake detector from frame folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset root containing train/val/test split folders.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/deepfake_detector.keras"),
        help="Where to save the best trained sequence model.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Square image size for each frame.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=12,
        help="Number of evenly sampled frames per video sequence.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for sequence training.",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=2,
        help="Warm-up epochs with the CNN backbone frozen.",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=4,
        help="Fine-tuning epochs with the top backbone layers unfrozen.",
    )
    parser.add_argument(
        "--learning-rate-head",
        type=float,
        default=1e-3,
        help="Learning rate for warm-up training.",
    )
    parser.add_argument(
        "--learning-rate-finetune",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--max-train-videos",
        type=int,
        default=None,
        help="Optional cap on the number of training videos.",
    )
    parser.add_argument(
        "--max-val-videos",
        type=int,
        default=None,
        help="Optional cap on the number of validation videos.",
    )
    parser.add_argument(
        "--max-test-videos",
        type=int,
        default=None,
        help="Optional cap on the number of test videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def get_split_dir(data_dir: Path, split: str) -> Path:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}. Expected {data_dir}/train, {data_dir}/val, and {data_dir}/test."
        )
    return split_dir


def classify_name(name: str):
    lowered = name.lower()
    if lowered.startswith("fake"):
        return 1
    if lowered.startswith("real"):
        return 0
    return None


def collect_video_samples(split_dir: Path):
    samples = []
    direct_children = [path for path in split_dir.iterdir() if path.is_dir()]
    child_names = {path.name.lower() for path in direct_children}

    if {"real", "fake"}.issubset(child_names):
        for class_dir in direct_children:
            label = 0 if class_dir.name.lower() == "real" else 1
            nested_dirs = [path for path in class_dir.iterdir() if path.is_dir()]
            if nested_dirs:
                for video_dir in nested_dirs:
                    if has_images(video_dir):
                        samples.append({"video_dir": video_dir, "label": label})
            else:
                image_paths = list_images(class_dir)
                if image_paths:
                    samples.append({"video_dir": class_dir, "label": label, "flat_images": image_paths})
    else:
        for video_dir in direct_children:
            label = classify_name(video_dir.name)
            if label is None:
                continue
            if has_images(video_dir):
                samples.append({"video_dir": video_dir, "label": label})

    if not samples:
        raise ValueError(
            f"No sequence samples found in {split_dir}. Expected split folders containing real/fake video folders or folders prefixed with real_/fake_."
        )

    return samples


def has_images(directory: Path) -> bool:
    return any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in directory.rglob("*"))


def list_images(directory: Path):
    return sorted(
        [
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def cap_samples(samples, max_items: int | None, seed: int):
    if max_items is None or len(samples) <= max_items:
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, max_items)


class VideoFrameSequence(tf.keras.utils.Sequence):
    def __init__(self, samples, batch_size: int, sequence_length: int, image_size: int, shuffle: bool, seed: int):
        self.samples = list(samples)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.indexes = list(range(len(self.samples)))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_samples = [self.samples[item] for item in batch_indexes]

        x = np.zeros(
            (len(batch_samples), self.sequence_length, self.image_size, self.image_size, 3),
            dtype=np.float32,
        )
        y = np.zeros((len(batch_samples), 1), dtype=np.float32)

        for position, sample in enumerate(batch_samples):
            x[position] = self.load_sequence(sample)
            y[position, 0] = sample["label"]
        return x, y

    def load_sequence(self, sample):
        image_paths = sample.get("flat_images")
        if image_paths is None:
            image_paths = list_images(sample["video_dir"])
        if not image_paths:
            raise ValueError(f"No frame images found in {sample['video_dir']}")

        sequence_paths = self.sample_paths(image_paths)
        frames = []
        for image_path in sequence_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame.astype(np.float32))

        if not frames:
            raise ValueError(f"Could not decode any frames from {sample['video_dir']}")

        while len(frames) < self.sequence_length:
            frames.append(frames[-1].copy())
        if len(frames) > self.sequence_length:
            frames = frames[:self.sequence_length]
        return np.stack(frames, axis=0)

    def sample_paths(self, image_paths):
        if len(image_paths) >= self.sequence_length:
            indexes = np.linspace(0, len(image_paths) - 1, num=self.sequence_length, dtype=int)
            return [image_paths[idx] for idx in indexes]
        paths = list(image_paths)
        while len(paths) < self.sequence_length:
            paths.append(paths[-1])
        return paths


def augmentation_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="sequence_augmentation",
    )


def build_model(sequence_length: int, image_size: int):
    inputs = tf.keras.Input(shape=(sequence_length, image_size, image_size, 3))
    x = tf.keras.layers.TimeDistributed(augmentation_layer())(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
        weights="imagenet",
    )
    backbone.trainable = False

    x = tf.keras.layers.TimeDistributed(backbone, name="frame_backbone")(x)
    x = tf.keras.layers.Masking()(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, dropout=0.2)
    )(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, backbone


def compile_model(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def load_saved_model(model_path: Path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
        },
        compile=False,
    )
    return model


def compute_class_weight(samples):
    positives = sum(sample["label"] for sample in samples)
    total = len(samples)
    negatives = total - positives
    if negatives == 0 or positives == 0:
        raise ValueError("Training split must contain both real and fake video samples.")
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def make_callbacks(model_out: Path):
    model_out.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_out),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]


def history_to_dict(history):
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def save_training_report(report_path: Path, payload: dict):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_samples(name: str, samples):
    positives = sum(sample["label"] for sample in samples)
    negatives = len(samples) - positives
    print(f"{name}: {len(samples)} videos | real={negatives} fake={positives}")


def main():
    args = parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = get_split_dir(args.data_dir, "train")
    val_dir = get_split_dir(args.data_dir, "val")
    test_dir = get_split_dir(args.data_dir, "test")

    print("Collecting video frame sequences...")
    train_samples = cap_samples(collect_video_samples(train_dir), args.max_train_videos, args.seed)
    val_samples = cap_samples(collect_video_samples(val_dir), args.max_val_videos, args.seed + 1)
    test_samples = cap_samples(collect_video_samples(test_dir), args.max_test_videos, args.seed + 2)

    summarize_samples("Train", train_samples)
    summarize_samples("Val", val_samples)
    summarize_samples("Test", test_samples)

    class_weight = compute_class_weight(train_samples)
    print(f"Class weights: {class_weight}")

    train_sequence = VideoFrameSequence(
        train_samples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        shuffle=True,
        seed=args.seed,
    )
    val_sequence = VideoFrameSequence(
        val_samples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        shuffle=False,
        seed=args.seed,
    )
    test_sequence = VideoFrameSequence(
        test_samples,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        shuffle=False,
        seed=args.seed,
    )

    model, backbone = build_model(sequence_length=args.sequence_length, image_size=args.image_size)
    compile_model(model, args.learning_rate_head)
    callbacks = make_callbacks(args.model_out)

    print("Training temporal head...")
    head_history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("Fine-tuning the top CNN layers with sequence learning...")
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    compile_model(model, args.learning_rate_finetune)
    finetune_history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=args.epochs_head + args.epochs_finetune,
        initial_epoch=args.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("Loading best checkpoint for evaluation...")
    best_model = load_saved_model(args.model_out)
    compile_model(best_model, args.learning_rate_finetune)
    test_metrics = best_model.evaluate(test_sequence, return_dict=True)
    test_metrics = {key: float(value) for key, value in test_metrics.items()}

    report = {
        "data_dir": str(args.data_dir),
        "model_out": str(args.model_out),
        "image_size": args.image_size,
        "sequence_length": args.sequence_length,
        "train_videos": len(train_samples),
        "val_videos": len(val_samples),
        "test_videos": len(test_samples),
        "class_weight": {str(key): float(value) for key, value in class_weight.items()},
        "head_history": history_to_dict(head_history),
        "finetune_history": history_to_dict(finetune_history),
        "test_metrics": test_metrics,
    }

    report_path = args.model_out.with_suffix(".metrics.json")
    save_training_report(report_path, report)

    print("Sequence training complete.")
    print(f"Saved best model to: {args.model_out}")
    print(f"Saved metrics to: {report_path}")
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
