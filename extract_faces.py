import argparse
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract face crops from real/fake videos for deepfake detector training."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing split/class video folders, for example data/raw/train/real/*.mp4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where extracted face images will be written.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=12,
        help="Number of frames to sample from each video.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output face crop size in pixels.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum detected face size in pixels.",
    )
    return parser.parse_args()


def load_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")
    return detector


def sample_frames(video_path: Path, frames_per_video: int):
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        indexes = np.linspace(0, total_frames - 1, num=min(total_frames, frames_per_video), dtype=int)
    else:
        indexes = np.arange(frames_per_video, dtype=int)

    frames = []
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
    capture.release()
    return frames


def extract_largest_face(frame, detector, image_size: int, min_face_size: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    margin = int(min(w, h) * 0.18)
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, frame.shape[1])
    y1 = min(y + h + margin, frame.shape[0])
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (image_size, image_size))
    return resized


def iter_videos(input_dir: Path):
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def process_video(video_path: Path, input_dir: Path, output_dir: Path, detector, frames_per_video: int, image_size: int, min_face_size: int):
    try:
        relative_parent = video_path.relative_to(input_dir).parent
    except ValueError:
        relative_parent = Path(".")

    destination = output_dir / relative_parent / video_path.stem
    destination.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for index, frame in enumerate(sample_frames(video_path, frames_per_video)):
        face = extract_largest_face(frame, detector, image_size=image_size, min_face_size=min_face_size)
        if face is None:
            continue
        file_path = destination / f"{video_path.stem}_{index:03d}.jpg"
        cv2.imwrite(str(file_path), face)
        extracted += 1
    return extracted


def main():
    args = parse_args()
    detector = load_detector()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    total_videos = 0
    total_faces = 0
    skipped = 0

    for video_path in iter_videos(args.input_dir):
        total_videos += 1
        faces = process_video(
            video_path=video_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            detector=detector,
            frames_per_video=args.frames_per_video,
            image_size=args.image_size,
            min_face_size=args.min_face_size,
        )
        if faces == 0:
            skipped += 1
        total_faces += faces
        print(f"{video_path}: extracted {faces} face crops")

    print(f"Processed videos: {total_videos}")
    print(f"Extracted faces: {total_faces}")
    print(f"Videos with no detected face crops: {skipped}")


if __name__ == "__main__":
    main()
