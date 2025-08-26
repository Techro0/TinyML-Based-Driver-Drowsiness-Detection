# TinyML Driver Drowsiness Detection (Open/Close Eyes)

A TinyML‑ready driver drowsiness detector trained on the **OACE (Open and Close Eyes)** dataset.
It classifies eye state (open vs. closed) from webcam frames and raises an alarm when both eyes
are closed for a configurable number of consecutive frames. The model is exported to **TensorFlow Lite (INT8)** for later deployment on microcontrollers with **TFLite Micro**.

<p align="center">
  <img src="Driver Drowsiness Detection.png" width="75%"><br/>
  <em>Diver Drowsiness Detection</em>
</p>

## Features
- Tiny depthwise‑separable CNN (64×64 grayscale) — quantization‑friendly
- Real‑time webcam inference with Haar face/eye detection (OpenCV)
- Drowsiness rule: both eyes closed for *N* consecutive frames → WAV alarm + MQTT alert
- Exports **INT8 TFLite** (`models/eye_state_int8.tflite`)
- Clear scripts for dataset prep, training, evaluation, export, and live inference

## Dataset
This project uses the **OACE (Open and Close Eyes)** dataset by *Muhammad Hanan Asghar* on Kaggle.

➡️ Dataset link: https://www.kaggle.com/datasets/muhammadhananasghar/oace-open-and-close-eyes-dataset

Please review the Kaggle page for license and usage terms before distributing the data.

## Project structure
```
tinyml-drowsy/
├─ data/
│  └─ OACE/                 # put dataset here: open/ and close/
├─ models/
│  ├─ best_model.keras
│  ├─ saved_model/
│  ├─ eye_state_int8.tflite
│  └─ label_map.txt
├─ scripts/
│  ├─ prepare_dataset.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ export_tflite.py
│  └─ infer_webcam.py
├─ alarms/
│  ├─ alarm.wav
│  └─ mqtt_demo_receiver.py
├─ notebooks/               # optional
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Setup
### 1) Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```
> Apple Silicon users: if needed use `tensorflow-macos` + `tensorflow-metal` instead of `tensorflow`.

### 3) Download the dataset
Dataset link: [OACE Open/Close Eyes on Kaggle](https://www.kaggle.com/datasets/muhammadhananasghar/oace-open-and-close-eyes-dataset)

**Kaggle CLI (recommended):**
```bash
kaggle datasets download -d muhammadhananasghar/oace-open-and-close-eyes-dataset -p data/
unzip data/oace-open-and-close-eyes-dataset.zip -d data/OACE
```
Make sure you get:
```
data/OACE/open/...
data/OACE/close/...
```
```bash
kaggle datasets download -d muhammadhananasghar/oace-open-and-close-eyes-dataset -p data/
unzip data/oace-open-and-close-eyes-dataset.zip -d data/OACE
```
Make sure you get:
```
data/OACE/open/...
data/OACE/close/...
```

### 4) Prepare split, train, evaluate, export
```bash
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/evaluate.py
python scripts/export_tflite.py
```

### 5) Run live inference (webcam + alarm + MQTT)
```bash
python scripts/infer_webcam.py
```
Press `q` or `Esc` to quit. Tune `CLOSED_FRAMES_THRESHOLD` (frames) inside `infer_webcam.py` to about 1.5–2.0 seconds of closure at your camera FPS.

### 6) MQTT demo (optional)
On another terminal or device:
```bash
python alarms/mqtt_demo_receiver.py
```
When drowsiness is detected, a message is published to `tinyml/drowsy/alert` via `test.mosquitto.org`.

## TinyML deployment (later)
1. Convert `models/eye_state_int8.tflite` to C array:
   ```bash
   xxd -i models/eye_state_int8.tflite > models/eye_state_int8.cc
   ```
2. Include in a TFLite‑Micro project (Arduino/ESP32/STM32). Use an arena of ~100–200KB (depends on platform).
3. Feed 64×64 grayscale eye crops (int8) with the same pre‑/post‑processing.

## Notes
- Lighting & cropping matter: if Haar misses eyes, try `haarcascade_eye_tree_eyeglasses.xml` or integrate landmarks (e.g., MediaPipe) to crop eyes.
- If classes are imbalanced, consider `class_weight` in training.

---

**Author:** Techro0 • MIT License
