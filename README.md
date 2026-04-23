# SignBridge — Real-time ASL Fingerspelling Recognition

SignBridge is a live American Sign Language (ASL) fingerspelling recognition system that runs entirely on a standard webcam. It translates hand signs into text in real time, character by character, and builds up a sentence buffer you can read back.

---

## Setup

### Requirements

- Python 3.9 or newer
- A webcam
- The [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) *(only needed if re-training)*

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the live application

Pre-trained models are already included in `models/`. Just run:

```bash
python app/app.py
```

The app opens a webcam window. Hold a hand sign steadily for ~1.5 seconds to confirm it. Confirmed letters appear in the sentence bar at the bottom.

| Key | Action |
|-----|--------|
| `SPACE` | Insert a space |
| `BACKSPACE` | Delete last character |
| `ENTER` | Clear the sentence |
| `Q` | Quit |

### Re-train from scratch (optional)

1. Download the Kaggle ASL Alphabet dataset and extract it so the folder structure is:
   ```
   data/raw/asl_alphabet_train/asl_alphabet_train/<A-Z>/
   ```
2. Run the training pipeline:
   ```bash
   python app/train.py
   ```
   This extracts hand keypoints, trains both a Random Forest and an MLP, selects the better one, and saves all models to `models/`.

---

## Project Structure

```
SignBridge/
├── app/
│   ├── app.py          # Live webcam application
│   └── train.py        # Training pipeline
├── data/
│   ├── keypoints_train.csv   # Extracted landmarks (53 486 samples)
│   ├── keypoints_val.csv     # Validation split   ( 6 686 samples)
│   ├── keypoints_test.csv    # Test split          ( 6 686 samples)
│   └── raw/                  # Raw ASL Alphabet images (Kaggle)
├── models/
│   ├── hand_landmarker.task  # MediaPipe pretrained hand detector
│   ├── static_model.pth      # Trained MLP classifier (PyTorch)
│   ├── static_model_rf.pkl   # Random Forest fallback (scikit-learn)
│   └── motion_model.pkl      # DTW templates for J and Z
└── requirements.txt
```

---

## The Problem

ASL fingerspelling (spelling words letter-by-letter using hand shapes) is the bridge between ASL and written English. Automating its recognition is genuinely hard for classical computer vision:

- **Viewpoint and scale sensitivity** — the same hand sign looks completely different depending on camera angle, hand size, and distance. Pixel-based classifiers (template matching, HOG + SVM) break immediately when these change.
- **Motion-dependent letters** — J and Z are not static poses; they require tracking a trajectory over time, which a single-frame approach cannot handle at all.
- **High inter-class similarity** — letters like A, E, M, N, S, and T all involve a closed fist with subtle finger position differences that are nearly indistinguishable from raw pixel features.

Neural networks solve the first two problems directly: a landmark detector gives pose-invariant 3-D joint coordinates, and a learned classifier on those coordinates generalises across viewpoints and hand sizes. The motion letters are handled with trajectory matching on the same landmark stream.

---

## Neural Network Design

### Component 1 — MediaPipe HandLandmarker (Pretrained)

**Architecture:** A two-stage pipeline combining a palm detector (BlazePalm, a lightweight MobileNet variant) followed by a hand landmark regression network. Together they output the 3-D (x, y, z) coordinates of 21 hand joints in a single forward pass.

**Why this is the hard part:** Localising 21 joints under occlusion, across arbitrary hand orientations, in real time on a CPU is exactly the problem that defeated classical methods. MediaPipe achieves this by training on ~30 000 real hands and ~60 000 synthetic renders. The synthetic data provides complete ground-truth annotation that would be prohibitively expensive to capture manually.

**Citation:** Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking*, CVPR Workshop 2020. Model weights are provided by Google under the Apache 2.0 licence.

**No fine-tuning was applied.** The pretrained landmark model generalises well to fingerspelling poses without modification.

### Component 2 — Static Letter Classifier (Trained from scratch)

**Input representation — tensor encoding:**
Each frame's 21 landmarks are converted to a 63-dimensional float32 tensor via the following normalisation:

1. Translate all coordinates so the wrist (landmark 0) is at the origin — removes absolute hand position from the input.
2. Divide by the maximum absolute coordinate value across all joints — scales the hand to fit within [−1, 1] regardless of distance from the camera.
3. Flatten: `[x₀, y₀, z₀, x₁, y₁, z₁, … x₂₀, y₂₀, z₂₀]` → shape `(63,)`.

This normalisation is the key step that makes the representation **viewpoint- and scale-invariant** — the same letter from a short distance and a long distance produces nearly identical tensors.

**Architecture — MLP (`StaticMLP`):**

```
Input (63)
  → Linear(63 → 256) → ReLU → BatchNorm → Dropout(0.30)
  → Linear(256 → 128) → ReLU → BatchNorm → Dropout(0.25)
  → Linear(128 → 64) → ReLU → Dropout(0.20)
  → Linear(64 → 26)
  → Softmax (at inference)
```

26 output classes: letters A–Y (excluding J and Z, which are motion-based) plus `del` and `space`.

**Training data:** The [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — ~87 000 images across 29 classes (A–Z + del, space, nothing). MediaPipe was run over every image to extract landmarks; frames where no hand was detected were discarded. This yielded **~66 858 labelled keypoint samples** split 80 / 10 / 10 into train / val / test.

**Training loop:**
- Optimiser: Adam (lr = 1e-3, weight decay = 1e-4)
- Loss: Cross-entropy with label smoothing (ε = 0.05) — reduces overconfidence on ambiguous letters
- Schedule: Cosine annealing over 100 epochs (min lr = 1e-5)
- Data augmentation applied per batch:
  - Random in-plane rotation ±20°
  - Random scale jitter ±8%
  - Gaussian coordinate noise (σ = 0.008)

A Random Forest (300 trees) was trained in parallel as a fallback; the model with the higher validation accuracy is saved as the primary classifier.

### Component 3 — Motion Classifier for J and Z (DTW)

J and Z involve moving the index fingertip along a specific trajectory. At runtime, a rolling 20-frame buffer of index fingertip positions (landmark 8, relative to the wrist) is maintained.

**Dynamic Time Warping (DTW)** compares this trajectory against hand-crafted reference templates for J and Z. DTW is appropriate here because it handles temporal stretching — a signer moving slowly or quickly produces the same match. The trajectory is normalised to its starting position and bounding span before comparison, making matching scale- and position-independent.

If the DTW confidence exceeds a threshold (0.45) and displacement is above a minimum (0.06), the motion classifier fires and overrides the static prediction.

---

## End-to-End Application Pipeline

```
Webcam frame (BGR)
        │
        ▼
  cv2.flip()          ← mirror so it feels natural
        │
        ▼
  BGR → RGB + mp.Image
        │
        ▼
  MediaPipe HandLandmarker.detect_for_video()
        │
        ├── No hand detected → clear buffers, display "?"
        │
        └── 21 NormalizedLandmark objects
                │
                ├─── normalize_landmarks() → float32 tensor (63,)
                │           │
                │           └─→ StaticMLP / RandomForest → (letter, confidence)
                │
                ├─── get_index_tip_xy() → (x, y) relative to wrist
                │           │
                │           └─→ motion_buffer (deque, 20 frames)
                │                       │
                │                       └─→ DTW vs J / Z templates → (letter, confidence)
                │
                └─── Fusion rule:
                        if motion_conf > 0.5  → use motion letter
                        elif static_conf < 0.35 → no prediction
                        else                   → use static letter
                              │
                              ▼
                   Majority-vote buffer (11 frames)
                              │
                              ▼
                   Hold counter (45 consecutive stable frames)
                              │
                              ▼
                   Confirmed letter → append to sentence
                              │
                              ▼
                   Cooldown (40 frames) → prevents duplicate
```

**Tensor encoding detail:** The model never sees raw pixels. The only data flowing into the classifiers is the 63-float landmark vector (or the 2-float index-tip position for the motion model). This keeps inference fast enough to run at 30 fps on a CPU.

---

## Deployment

The application runs locally. No GPU is required — MediaPipe and the MLP both run efficiently on CPU at 30 fps.

**Platform:** Windows / macOS / Linux with Python 3.9+, `pip`, and a USB or built-in webcam.

**Runtime dependencies:** see `requirements.txt`. Key libraries: `mediapipe`, `torch`, `opencv-python`, `scikit-learn`.
