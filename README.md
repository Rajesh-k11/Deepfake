
# ForgeSight Deepfake Detection Studio

ForgeSight is a customized Streamlit app for deepfake triage and model-backed detection. It improves the original repository by replacing the invalid untrained inference flow with a real analysis pipeline that:

- samples video frames evenly
- detects and crops faces
- computes forensic artifact metrics such as blockiness, noise mismatch, blur variance, and temporal instability
- shows frame evidence and explainable findings
- optionally loads a trained TensorFlow checkpoint for higher accuracy

## What Changed

The original app created a brand-new LSTM at prediction time, which means it could not produce meaningful real/fake results. This version fixes that by making the detector honest and extensible:

- heuristic mode now provides explainable deepfake risk scoring when no trained model is present
- model mode automatically loads `models/deepfake_detector.keras` or `models/deepfake_detector.h5`
- the UI is redesigned for clearer verdicts, evidence tabs, metrics, and next-step guidance
- uploaded videos are stored in temporary files and cleaned up automatically

## Project Structure

```text
deepfake-detection/
|-- app.py
|-- README.md
|-- requirements.txt
|-- deepfake-model.json
|-- models/
|   |-- deepfake_detector.keras   # optional, add your trained checkpoint here
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ParivalavanIT/deepfake-detection.git
   cd deepfake-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run The App

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a face-forward video, and review:

- verdict and confidence
- deepfake vs authentic risk balance
- artifact-based findings
- frame previews
- analysis metrics

## Accuracy Notes

This repository is now ready for a real model, but benchmarked accuracy depends on the checkpoint you provide.

- Without a checkpoint, the app runs in heuristic mode for forensic triage only.
- With a trained checkpoint, the app uses TensorFlow inference and becomes much more useful.
- For production-quality accuracy, train and validate on labeled datasets such as FaceForensics++, DFDC, Celeb-DF, and DeepFakeDetection.
- Measure precision, recall, F1, ROC-AUC, and false-positive rate on a held-out validation set.

## Adding A Trained Model

Place one of these files in the `models/` directory:

- `models/deepfake_detector.keras`
- `models/deepfake_detector.h5`

The app will detect it automatically and switch from heuristic mode to TensorFlow model mode.

## Training Your Own Model

The repo now includes a basic but real training pipeline:

- [extract_faces.py](/D:/DeepFake/deepfake-detection/extract_faces.py:1) converts raw videos into face crops
- [train_model.py](/D:/DeepFake/deepfake-detection/train_model.py:1) trains an EfficientNet-based binary classifier and saves `models/deepfake_detector.keras`
- [train_sequence_model.py](/D:/DeepFake/deepfake-detection/train_sequence_model.py:1) trains a temporal model over multiple frames per video folder for better video accuracy

### 1. Arrange the dataset

Put your data in this structure:

```text
data/
|-- raw/
|   |-- train/
|   |   |-- real/
|   |   |-- fake/
|   |-- val/
|   |   |-- real/
|   |   |-- fake/
|   |-- test/
|   |   |-- real/
|   |   |-- fake/
```

Each folder can contain videos such as `.mp4`, `.avi`, `.mov`, `.mkv`, `.mpeg`, or `.webm`.

### 2. Extract face crops

```bash
python extract_faces.py --input-dir data/raw --output-dir data/processed --frames-per-video 12 --image-size 224
```

That will create:

```text
data/processed/
|-- train/
|   |-- real/
|   |-- fake/
|-- val/
|   |-- real/
|   |-- fake/
|-- test/
|   |-- real/
|   |-- fake/
```

### 3. Train the model

```bash
python train_model.py --data-dir data/processed --model-out models/deepfake_detector.keras
```

Optional tuning example:

```bash
python train_model.py --data-dir data/processed --model-out models/deepfake_detector.keras --batch-size 16 --epochs-head 5 --epochs-finetune 12
```

The trainer will:

- load face images from `train`, `val`, and `test`
- fine-tune `EfficientNetB0` pretrained on ImageNet
- use augmentation, class weighting, early stopping, and best-checkpoint saving
- save a metrics report next to the model, for example `models/deepfake_detector.metrics.json`

### 3b. Train the stronger sequence model

If your dataset is organized as frame folders per video, use the temporal trainer instead. This is a better fit for video deepfake detection because it learns across multiple frames from the same video.

Example with extracted face folders:

```bash
python train_sequence_model.py --data-dir data --model-out models/deepfake_detector.keras --sequence-length 12 --batch-size 4
```

Example with a smaller sampled dataset:

```bash
python train_sequence_model.py --data-dir data_small --model-out models/deepfake_detector.keras --sequence-length 12 --batch-size 4 --epochs-head 2 --epochs-finetune 4
```

The sequence trainer will:

- treat each video folder as one labeled sample
- evenly sample multiple frames from that folder
- extract per-frame features with `EfficientNetB0`
- learn temporal patterns with bidirectional `LSTM` and `GRU` layers
- save a sequence-ready checkpoint that the app can load directly

### 4. Run the app

After training finishes:

```bash
streamlit run app.py
```

The app will automatically load `models/deepfake_detector.keras`.

## How To Improve Accuracy

If your goal is stronger accuracy rather than just “a working model,” these matter the most:

- Use a large and balanced dataset. More high-quality real/fake faces helps more than small architecture tweaks.
- Keep train, val, and test identities separated. If the same person leaks across splits, accuracy will look falsely high.
- Mix datasets such as FaceForensics++, DFDC, Celeb-DF, and DeepFakeDetection.
- Use good face crops. Poor detection and badly aligned faces will hurt accuracy before training even starts.
- Prefer `train_sequence_model.py` over `train_model.py` for video-first detection work.
- Tune the decision threshold on validation data instead of assuming `0.50` is best.
- Track `precision`, `recall`, `AUC`, and false positives, not just accuracy.
- Prefer videos with clear frontal faces and consistent lighting.

## Stronger Upgrades After This

If you want to go beyond the included baseline trainer, the next upgrades with the biggest payoff are:

- temporal sequence modeling over face frames instead of frame-only classification
- stronger backbones such as EfficientNetV2 or Xception
- landmark or frequency-domain features as an additional branch
- threshold calibration and validation reporting per dataset
- hard-negative mining for videos that look real but are fake

## Contact

Original repository contact: `parivalavan2345@gmail.com`
