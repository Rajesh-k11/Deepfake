import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    import tensorflow as tf
except Exception:
    tf = None


APP_NAME = "ForgeSight"
FRAME_SIZE = 224
MAX_SAMPLED_FRAMES = 24
MIN_FACE_FRAMES = 6
MODEL_CANDIDATES = (
    Path("models/deepfake_detector.keras"),
    Path("models/deepfake_detector.h5"),
)


st.set_page_config(page_title=APP_NAME, page_icon="AI", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f3efe7;
                --panel: #fbf8f3;
                --panel-strong: #f0e7da;
                --sidebar: #1d2733;
                --sidebar-2: #243240;
                --ink: #18222d;
                --muted: #5f6c78;
                --line: rgba(24, 34, 45, 0.08);
                --accent: #b85c38;
                --accent-soft: rgba(184, 92, 56, 0.12);
                --teal: #1f6b6a;
                --teal-soft: rgba(31, 107, 106, 0.10);
                --blue-soft: rgba(72, 109, 146, 0.12);
                --gold-soft: rgba(181, 145, 58, 0.16);
            }
            .stApp {
                background:
                    linear-gradient(rgba(255,255,255,0.38) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.38) 1px, transparent 1px),
                    radial-gradient(circle at top left, rgba(184, 92, 56, 0.12), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(31, 107, 106, 0.10), transparent 24%),
                    linear-gradient(180deg, #f7f3ec 0%, var(--bg) 100%);
                background-size: 32px 32px, 32px 32px, auto, auto, auto;
                color: var(--ink);
            }
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, var(--sidebar) 0%, var(--sidebar-2) 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
            }
            [data-testid="stSidebar"] * {
                color: #f4efe8;
            }
            [data-testid="stSidebar"] code {
                color: #ffd8c6 !important;
                background: rgba(255,255,255,0.08);
                word-break: break-word;
                overflow-wrap: anywhere;
                white-space: normal;
            }
            .hero {
                position: relative;
                overflow: hidden;
                background: linear-gradient(145deg, rgba(251,248,243,0.98), rgba(240,231,218,0.94));
                border: 1px solid var(--line);
                border-radius: 28px;
                padding: 34px;
                box-shadow: 0 24px 40px rgba(24, 34, 45, 0.08);
                margin-bottom: 20px;
            }
            .hero::after {
                content: "";
                position: absolute;
                inset: auto -40px -60px auto;
                width: 220px;
                height: 220px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(184,92,56,0.16), rgba(184,92,56,0));
            }
            .hero h1 {
                margin: 0 0 8px 0;
                font-size: 3rem;
                line-height: 0.98;
                max-width: 800px;
                color: var(--ink);
            }
            .hero p {
                margin: 0;
                color: var(--muted);
                font-size: 1.04rem;
                line-height: 1.7;
                max-width: 920px;
            }
            .metric-card {
                background: rgba(251, 248, 243, 0.95);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 18px;
                margin-bottom: 12px;
                box-shadow: 0 10px 24px rgba(24, 34, 45, 0.05);
            }
            .metric-label {
                font-size: 0.85rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.10em;
            }
            .metric-value {
                font-size: 1.65rem;
                font-weight: 700;
                color: var(--ink);
            }
            .pill {
                display: inline-block;
                padding: 8px 14px;
                border-radius: 999px;
                background: rgba(24, 34, 45, 0.06);
                color: var(--accent);
                font-weight: 600;
                font-size: 0.82rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .notice-card {
                border-radius: 18px;
                padding: 16px 18px;
                border: 1px solid transparent;
                margin: 10px 0 16px 0;
                line-height: 1.6;
            }
            .notice-card strong {
                display: block;
                margin-bottom: 4px;
                font-size: 0.96rem;
            }
            .notice-info {
                background: var(--blue-soft);
                border-color: rgba(72, 109, 146, 0.16);
                color: #24415c;
            }
            .notice-success {
                background: var(--teal-soft);
                border-color: rgba(31, 107, 106, 0.16);
                color: #134f4d;
            }
            .notice-warn {
                background: var(--gold-soft);
                border-color: rgba(181, 145, 58, 0.18);
                color: #6a5517;
            }
            .notice-danger {
                background: rgba(170, 69, 57, 0.10);
                border-color: rgba(170, 69, 57, 0.18);
                color: #7a2f26;
            }
            .sidebar-panel {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 16px;
                margin-bottom: 16px;
            }
            .sidebar-panel h4 {
                margin: 0 0 8px 0;
                font-size: 0.95rem;
            }
            .sidebar-panel p, .sidebar-panel li {
                color: rgba(244, 239, 232, 0.88);
            }
            .sidebar-path {
                display: inline-block;
                margin-top: 6px;
                padding: 6px 8px;
                border-radius: 10px;
                background: rgba(255,255,255,0.08);
                color: #ffd8c6;
                font-family: Consolas, "Courier New", monospace;
                font-size: 0.88rem;
                line-height: 1.5;
                word-break: break-word;
                overflow-wrap: anywhere;
            }
            [data-testid="stFileUploader"] {
                background: rgba(251,248,243,0.94);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 10px;
                box-shadow: 0 10px 24px rgba(24, 34, 45, 0.05);
            }
            [data-testid="stFileUploader"] label,
            [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] span {
                color: var(--ink) !important;
            }
            [data-testid="stFileUploader"] section {
                background: transparent;
            }
            [data-testid="stFileUploaderDropzone"] {
                background: linear-gradient(180deg, rgba(251,248,243,0.95), rgba(240,231,218,0.9));
                border: 1.5px dashed rgba(184,92,56,0.28);
                border-radius: 18px;
            }
            [data-testid="stFileUploaderDropzone"] * {
                color: var(--ink) !important;
            }
            [data-testid="stFileUploaderDropzone"] button {
                background: linear-gradient(135deg, #b85c38 0%, #9f4c2d 100%) !important;
                color: #fff8f2 !important;
                border: 1px solid rgba(159, 76, 45, 0.45) !important;
                border-radius: 14px !important;
                box-shadow: 0 10px 20px rgba(184, 92, 56, 0.22) !important;
                font-weight: 700 !important;
            }
            [data-testid="stFileUploaderDropzone"] button:hover {
                background: linear-gradient(135deg, #c66742 0%, #ab5231 100%) !important;
                color: #ffffff !important;
                border-color: rgba(171, 82, 49, 0.55) !important;
            }
            [data-testid="stFileUploaderDropzone"] button:focus,
            [data-testid="stFileUploaderDropzone"] button:focus-visible {
                outline: none !important;
                box-shadow: 0 0 0 4px rgba(184, 92, 56, 0.18), 0 10px 20px rgba(184, 92, 56, 0.24) !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background: rgba(251,248,243,0.92);
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 8px 18px;
                color: var(--muted);
            }
            .stTabs [aria-selected="true"] {
                background: rgba(184,92,56,0.10) !important;
                border-color: rgba(184,92,56,0.22) !important;
                color: var(--accent) !important;
            }
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #1f6b6a 0%, #b85c38 100%);
            }
            [data-testid="stNotification"], .stAlert {
                background: rgba(251,248,243,0.90) !important;
                border: 1px solid var(--line) !important;
                color: var(--ink) !important;
                border-radius: 18px !important;
            }
            @media (max-width: 900px) {
                .hero {
                    padding: 24px;
                }
                .hero h1 {
                    font-size: 2.3rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


@st.cache_resource
def load_optional_model():
    if tf is None:
        return None, None

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model = tf.keras.models.load_model(
                candidate,
                custom_objects={
                    "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
                },
                compile=False,
            )
            return model, candidate
    return None, None


@st.cache_data
def load_model_report(model_path: str | None):
    if not model_path:
        return None

    report_path = Path(model_path).with_suffix(".metrics.json")
    if not report_path.exists():
        return None

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def sample_video_frames(video_path: str, max_frames: int = MAX_SAMPLED_FRAMES):
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0

    if total_frames > 0:
        frame_indexes = np.linspace(0, total_frames - 1, num=min(total_frames, max_frames), dtype=int)
    else:
        frame_indexes = np.arange(max_frames, dtype=int)

    frames = []
    for index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()
    return frames, {"total_frames": total_frames, "fps": fps}


def extract_face(frame: np.ndarray, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return None, 0.0

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    margin = int(min(w, h) * 0.18)
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, frame.shape[1])
    y1 = min(y + h + margin, frame.shape[0])
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0

    coverage = (w * h) / float(frame.shape[0] * frame.shape[1])
    resized = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))
    return resized, coverage


def blockiness_score(gray: np.ndarray, block: int = 8) -> float:
    vertical_boundaries = np.arange(block, gray.shape[1], block)
    horizontal_boundaries = np.arange(block, gray.shape[0], block)

    vertical_diff = [
        np.mean(np.abs(gray[:, idx].astype(np.float32) - gray[:, idx - 1].astype(np.float32)))
        for idx in vertical_boundaries
        if idx < gray.shape[1]
    ]
    horizontal_diff = [
        np.mean(np.abs(gray[idx, :].astype(np.float32) - gray[idx - 1, :].astype(np.float32)))
        for idx in horizontal_boundaries
        if idx < gray.shape[0]
    ]

    values = vertical_diff + horizontal_diff
    return float(np.mean(values)) if values else 0.0


def compute_face_metrics(face_frames, coverage_scores):
    blur_scores = []
    noise_scores = []
    block_scores = []
    edge_scores = []
    temporal_scores = []

    previous_face = None
    for face in face_frames:
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        noise_scores.append(float(np.std(gray.astype(np.float32) - denoised.astype(np.float32))))
        block_scores.append(blockiness_score(gray))
        edge_scores.append(float(np.mean(cv2.Canny(gray, 100, 200)) / 255.0))

        normalized = gray.astype(np.float32) / 255.0
        if previous_face is not None:
            temporal_scores.append(float(np.mean(np.abs(normalized - previous_face))))
        previous_face = normalized

    return {
        "face_frames": float(len(face_frames)),
        "mean_face_coverage": float(np.mean(coverage_scores)) if coverage_scores else 0.0,
        "blur": float(np.mean(blur_scores)) if blur_scores else 0.0,
        "noise": float(np.mean(noise_scores)) if noise_scores else 0.0,
        "blockiness": float(np.mean(block_scores)) if block_scores else 0.0,
        "edge_density": float(np.mean(edge_scores)) if edge_scores else 0.0,
        "temporal_instability": float(np.mean(temporal_scores)) if temporal_scores else 0.0,
    }


def normalize_metric(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def heuristic_assessment(metrics):
    artifact_signal = normalize_metric(metrics["blockiness"], 8.0, 22.0)
    temporal_signal = normalize_metric(metrics["temporal_instability"], 0.05, 0.20)
    noise_signal = normalize_metric(metrics["noise"], 7.0, 26.0)
    low_blur_penalty = 1.0 - normalize_metric(metrics["blur"], 60.0, 230.0)
    weak_face_penalty = 1.0 - normalize_metric(metrics["face_frames"], MIN_FACE_FRAMES, MAX_SAMPLED_FRAMES * 0.7)

    deepfake_score = (
        0.34 * artifact_signal
        + 0.26 * temporal_signal
        + 0.18 * noise_signal
        + 0.12 * low_blur_penalty
        + 0.10 * weak_face_penalty
    )
    deepfake_score = float(np.clip(deepfake_score, 0.0, 1.0))

    label = "Likely Deepfake" if deepfake_score >= 0.55 else "Likely Authentic"
    confidence = float(np.clip(0.52 + abs(deepfake_score - 0.5) * 0.9, 0.5, 0.96))

    drivers = {
        "Compression artifacts": artifact_signal,
        "Temporal instability": temporal_signal,
        "Sensor/noise mismatch": noise_signal,
        "Low natural detail": low_blur_penalty,
        "Limited face evidence": weak_face_penalty,
    }
    sorted_drivers = sorted(drivers.items(), key=lambda item: item[1], reverse=True)

    findings = []
    for name, score in sorted_drivers[:3]:
        if score < 0.35:
            continue
        findings.append(f"{name} scored higher than expected for a natural face sequence.")

    if not findings:
        findings.append("The sampled face frames looked relatively consistent across compression, texture, and motion cues.")

    return {
        "label": label,
        "confidence": confidence,
        "score_fake": deepfake_score,
        "score_real": 1.0 - deepfake_score,
        "mode": "Heuristic artifact analysis",
        "findings": findings,
    }


def prepare_model_input(face_frames, model) -> np.ndarray:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    frames = np.array(face_frames, dtype=np.float32) / 255.0

    if len(input_shape) == 5:
        timesteps, height, width = input_shape[1], input_shape[2], input_shape[3]
        timesteps = timesteps or min(len(frames), MAX_SAMPLED_FRAMES)
        height = height or FRAME_SIZE
        width = width or FRAME_SIZE

        resized = np.array([cv2.resize(frame, (width, height)) for frame in frames], dtype=np.float32)
        if len(resized) < timesteps:
            padding = np.repeat(resized[-1][None, ...], timesteps - len(resized), axis=0)
            resized = np.concatenate([resized, padding], axis=0)
        else:
            resized = resized[:timesteps]
        return resized[None, ...]

    if len(input_shape) == 4:
        height, width = input_shape[1], input_shape[2]
        height = height or FRAME_SIZE
        width = width or FRAME_SIZE
        return np.array([cv2.resize(frame, (width, height)) for frame in frames], dtype=np.float32)

    raise ValueError("Unsupported TensorFlow model input shape.")


def model_assessment(face_frames, model, model_report=None):
    model_input = prepare_model_input(face_frames, model)
    predictions = model.predict(model_input, verbose=0)

    if predictions.ndim == 2 and predictions.shape[-1] >= 2:
        fake_prob = float(np.mean(predictions[:, -1]))
    elif predictions.ndim == 2 and predictions.shape[-1] == 1:
        fake_prob = float(np.mean(predictions[:, 0]))
    else:
        fake_prob = float(np.ravel(predictions)[0])

    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
    label = "AI-generated" if fake_prob >= 0.5 else "Authentic"

    test_metrics = (model_report or {}).get("test_metrics", {})
    benchmark_auc = test_metrics.get("auc")
    if benchmark_auc is None:
        benchmark_auc = test_metrics.get("AUC")

    margin = abs(fake_prob - 0.5)
    confidence = 0.55 + margin * 0.9

    if benchmark_auc is not None:
        reliability = float(np.clip((float(benchmark_auc) - 0.5) / 0.5, 0.0, 1.0))
        confidence = 0.5 + margin * 0.9 * reliability
    confidence = float(np.clip(confidence, 0.5, 0.99))

    findings = [
        "Prediction produced by the loaded TensorFlow checkpoint.",
        "Class labels are shown as AI-generated vs Authentic for demo clarity.",
    ]

    if benchmark_auc is not None:
        findings.append(
            f"Saved checkpoint benchmark: test AUC {float(benchmark_auc):.3f}. Treat this as a prototype signal, not a final forensic decision."
        )
    else:
        findings.append(
            "No sidecar validation report was found for this checkpoint, so its reliability is unknown."
        )

    findings.append(
        "Validate the checkpoint on a held-out benchmark before using it in production."
    )

    return {
        "label": label,
        "confidence": confidence,
        "score_fake": fake_prob,
        "score_real": 1.0 - fake_prob,
        "mode": "TensorFlow model",
        "benchmark_auc": float(benchmark_auc) if benchmark_auc is not None else None,
        "findings": findings,
    }


def analyze_video(video_path: str):
    frames, video_info = sample_video_frames(video_path)
    detector = load_face_detector()

    face_frames = []
    coverage_scores = []
    preview_frames = []

    for frame in frames:
        face_crop, coverage = extract_face(frame, detector)
        if face_crop is not None:
            face_frames.append(face_crop)
            coverage_scores.append(coverage)
            if len(preview_frames) < 6:
                preview_frames.append(face_crop)

    if not face_frames:
        return {
            "error": "No face was detected in the sampled frames. Try a clearer face-forward video.",
            "video_info": video_info,
        }

    metrics = compute_face_metrics(face_frames, coverage_scores)
    model, model_path = load_optional_model()

    if model is not None and len(face_frames) >= MIN_FACE_FRAMES:
        model_report = load_model_report(str(model_path))
        assessment = model_assessment(face_frames, model, model_report=model_report)
        assessment["model_path"] = str(model_path)
    else:
        assessment = heuristic_assessment(metrics)
        assessment["model_path"] = None

    return {
        "video_info": video_info,
        "metrics": metrics,
        "assessment": assessment,
        "preview_frames": preview_frames,
    }


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(title: str, body: str, tone: str = "info") -> None:
    st.markdown(
        f"""
        <div class="notice-card notice-{tone}">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(model_path):
    mode_label = "Model-ready detector"
    if model_path:
        mode_label = f"Checkpoint loaded: {Path(model_path).name}"

    st.markdown(
        f"""
        <div class="hero">
            <span class="pill">{mode_label}</span>
            <h1>{APP_NAME} Forensic Video Lab</h1>
            <p>
                Review face evidence, artifact signals, and model confidence in one place. The interface is tuned for
                practical inspection, and it can switch from heuristic analysis to a trained TensorFlow checkpoint as soon
                as you add one in <code>models/</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_path):
    with st.sidebar:
        st.subheader("Detection Setup")
        if model_path:
            st.markdown(
                f"""
                <div class="sidebar-panel">
                    <h4>Model Status</h4>
                    <p>Using trained checkpoint <code>{Path(model_path).name}</code>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif tf is None:
            st.markdown(
                """
                <div class="sidebar-panel">
                    <h4>Model Status</h4>
                    <p>TensorFlow is unavailable, so the app is currently running in heuristic mode.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-panel">
                    <h4>Model Status</h4>
                    <p>No trained checkpoint found yet.</p>
                    <p>Add this file for stronger accuracy:</p>
                    <div class="sidebar-path">models/deepfake_detector.keras</div>
                    <p style="margin-top:10px;">You can also use <code>.h5</code>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="sidebar-panel">
                <h4>What Improves Accuracy</h4>
                <ul>
                    <li>Use a frontal face video with stable lighting</li>
                    <li>Upload at least 3 to 5 seconds of footage</li>
                    <li>Add a trained TensorFlow detector checkpoint</li>
                    <li>Validate thresholds on your own dataset before production use</li>
                </ul>
            </div>
            """
            ,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sidebar-panel">
                <h4>Quick Test</h4>
                <p>Use one of the local sample videos:</p>
                <div class="sidebar-path">samples/sample_video.mp4</div>
                <div class="sidebar-path" style="margin-top:8px;">samples/sample_5s.mp4</div>
                <div class="sidebar-path" style="margin-top:8px;">samples/sample_10s.mp4</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


inject_styles()
_, model_path = load_optional_model()
render_header(str(model_path) if model_path else None)
render_sidebar(str(model_path) if model_path else None)

render_notice(
    "Decision support, not a final verdict",
    "Heuristic mode is useful for triage, but production-grade accuracy still depends on a trained and validated checkpoint.",
    tone="warn",
)

uploaded_file = st.file_uploader(
    "Upload a video for analysis",
    type=["mp4", "avi", "mov", "mkv", "mpeg", "webm"],
    help="Best results come from videos where the main face is clearly visible in multiple frames.",
)

if uploaded_file is not None:
    temp_path = save_uploaded_video(uploaded_file)
    try:
        with st.spinner("Scanning video frames and extracting face evidence..."):
            results = analyze_video(temp_path)

        if "error" in results:
            render_notice("Analysis could not start", results["error"], tone="danger")
        else:
            assessment = results["assessment"]
            metrics = results["metrics"]
            video_info = results["video_info"]

            left, center, right = st.columns(3)
            with left:
                metric_card("Predicted Class", assessment["label"])
            with center:
                metric_card("Confidence", f"{assessment['confidence'] * 100:.1f}%")
            with right:
                metric_card("Detection Mode", assessment["mode"])

            st.subheader("Risk Balance")
            st.progress(float(assessment["score_fake"]))
            st.caption(
                f"AI-generated signal: {assessment['score_fake'] * 100:.1f}% | "
                f"Authentic signal: {assessment['score_real'] * 100:.1f}%"
            )

            evidence_tab, metrics_tab, frames_tab, guide_tab = st.tabs(
                ["Evidence", "Metrics", "Frames", "How to Improve Accuracy"]
            )

            with evidence_tab:
                if assessment["label"] == "AI-generated":
                    render_notice(
                        "Predicted class: AI-generated",
                        "The strongest combined signals point toward synthetic or edited content.",
                        tone="danger",
                    )
                else:
                    render_notice(
                        "Predicted class: Authentic",
                        "The sampled face dynamics and artifacts look more consistent with authentic footage.",
                        tone="success",
                    )

                st.markdown("### Why the app leaned this way")
                for finding in assessment["findings"]:
                    st.write(f"- {finding}")

                if assessment.get("model_path"):
                    st.caption(f"Checkpoint used: `{assessment['model_path']}`")
                if assessment.get("benchmark_auc") is not None:
                    st.caption(f"Saved benchmark AUC: `{assessment['benchmark_auc']:.3f}`")

            with metrics_tab:
                metric_columns = st.columns(4)
                display_metrics = [
                    ("Face frames", f"{int(metrics['face_frames'])}"),
                    ("Face coverage", f"{metrics['mean_face_coverage'] * 100:.1f}%"),
                    ("Blur variance", f"{metrics['blur']:.1f}"),
                    ("Blockiness", f"{metrics['blockiness']:.2f}"),
                    ("Noise mismatch", f"{metrics['noise']:.2f}"),
                    ("Edge density", f"{metrics['edge_density']:.2f}"),
                    ("Temporal instability", f"{metrics['temporal_instability']:.3f}"),
                    ("Video FPS", f"{video_info['fps']:.2f}" if video_info["fps"] else "Unknown"),
                ]
                for index, (label, value) in enumerate(display_metrics):
                    with metric_columns[index % 4]:
                        metric_card(label, value)

            with frames_tab:
                st.markdown("### Sampled face crops")
                if results["preview_frames"]:
                    preview_columns = st.columns(min(3, len(results["preview_frames"])))
                    for index, frame in enumerate(results["preview_frames"]):
                        with preview_columns[index % len(preview_columns)]:
                            st.image(frame, caption=f"Face sample {index + 1}", use_container_width=True)
                else:
                    st.info("No preview frames were available.")

            with guide_tab:
                st.markdown(
                    """
                    **Recommended next steps**
                    - Add a real checkpoint at `models/deepfake_detector.keras` or `models/deepfake_detector.h5`
                    - Train or fine-tune on datasets such as FaceForensics++, DFDC, Celeb-DF, and DeepFakeDetection
                    - Measure precision, recall, F1, ROC-AUC, and false-positive rate on a held-out validation split
                    - Calibrate the decision threshold instead of assuming `0.50` is ideal for your use case
                    - Keep the heuristic report as an explainability layer even after you add a trained model
                    """
                )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    render_notice(
        "Ready for analysis",
        "Upload a face-forward video to begin the forensic scan.",
        tone="info",
    )
