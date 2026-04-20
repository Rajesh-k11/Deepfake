import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tensorflow as tf


IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a deepfake detector checkpoint for ForgeSight."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dataset root containing train/val/test folders with real and fake subfolders.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/deepfake_detector.keras"),
        help="Where to save the best trained model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=5,
        help="Warm-up epochs with the backbone frozen.",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=10,
        help="Fine-tuning epochs with top backbone layers unfrozen.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=256,
        help="Shuffle buffer size for training data.",
    )
    return parser.parse_args()


def get_split_dir(data_dir: Path, split: str) -> Path:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing split directory: {split_dir}. Expected {data_dir}/train, {data_dir}/val, and {data_dir}/test."
        )
    return split_dir


def build_dataset(split_dir: Path, batch_size: int, seed: int, shuffle: bool):
    class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    simple_layout = {path.name.lower() for path in class_dirs}

    if {"real", "fake"}.issubset(simple_layout):
        return tf.keras.utils.image_dataset_from_directory(
            split_dir,
            labels="inferred",
            label_mode="binary",
            batch_size=batch_size,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            shuffle=shuffle,
            seed=seed,
        )

    image_paths = []
    labels = []
    for class_dir in class_dirs:
        folder_name = class_dir.name.lower()
        if folder_name.startswith("fake"):
            label = 1
        elif folder_name.startswith("real"):
            label = 0
        else:
            continue

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(str(image_path))
                labels.append(label)

    if not image_paths:
        raise ValueError(
            f"No labeled images found in {split_dir}. Expected either real/fake folders or folders prefixed with real_/fake_."
        )

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        path_ds = path_ds.shuffle(min(len(image_paths), 2048), seed=seed, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3, try_recover_truncated=True)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        label = tf.expand_dims(label, axis=-1)
        return image, label

    dataset = path_ds.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def optimize_dataset(dataset, training: bool, shuffle_buffer: int):
    if training:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.prefetch(1)
    return dataset


def augmentation_layers():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    augment = augmentation_layers()(inputs)
    preprocess = tf.keras.applications.efficientnet.preprocess_input(augment)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
        weights="imagenet",
    )
    backbone.trainable = False
    features = backbone(preprocess, training=False)
    x = tf.keras.layers.Dropout(0.35)(features)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
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


def count_labels(dataset):
    positives = 0
    total = 0
    for _, labels in dataset:
        label_values = labels.numpy().reshape(-1)
        positives += int(label_values.sum())
        total += int(label_values.shape[0])
    negatives = total - positives
    return negatives, positives


def compute_class_weight(train_dataset):
    negatives, positives = count_labels(train_dataset)
    if negatives == 0 or positives == 0:
        raise ValueError("Training split must contain both real and fake examples.")
    total = negatives + positives
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
            patience=4,
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


def main():
    args = parse_args()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(args.seed)

    train_dir = get_split_dir(args.data_dir, "train")
    val_dir = get_split_dir(args.data_dir, "val")
    test_dir = get_split_dir(args.data_dir, "test")

    print("Loading datasets...")
    train_dataset = optimize_dataset(
        build_dataset(train_dir, args.batch_size, args.seed, shuffle=True),
        training=True,
        shuffle_buffer=args.shuffle_buffer,
    )
    val_dataset = optimize_dataset(
        build_dataset(val_dir, args.batch_size, args.seed, shuffle=False),
        training=False,
        shuffle_buffer=args.shuffle_buffer,
    )
    test_dataset = optimize_dataset(
        build_dataset(test_dir, args.batch_size, args.seed, shuffle=False),
        training=False,
        shuffle_buffer=args.shuffle_buffer,
    )

    print("Computing class weights...")
    class_weight = compute_class_weight(train_dataset)
    print(f"Class weights: {class_weight}")

    model, backbone = build_model()
    compile_model(model, args.learning_rate_head)
    callbacks = make_callbacks(args.model_out)

    print("Training classifier head...")
    head_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("Fine-tuning top backbone layers...")
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    compile_model(model, args.learning_rate_finetune)
    finetune_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs_head + args.epochs_finetune,
        initial_epoch=args.epochs_head,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("Loading best checkpoint for evaluation...")
    best_model = tf.keras.models.load_model(args.model_out)
    test_metrics = best_model.evaluate(test_dataset, return_dict=True)
    test_metrics = {key: float(value) for key, value in test_metrics.items()}

    report = {
        "data_dir": str(args.data_dir),
        "model_out": str(args.model_out),
        "class_weight": {str(key): float(value) for key, value in class_weight.items()},
        "head_history": history_to_dict(head_history),
        "finetune_history": history_to_dict(finetune_history),
        "test_metrics": test_metrics,
    }

    report_path = args.model_out.with_suffix(".metrics.json")
    save_training_report(report_path, report)

    print("Training complete.")
    print(f"Saved best model to: {args.model_out}")
    print(f"Saved metrics to: {report_path}")
    print("Test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
