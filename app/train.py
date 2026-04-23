#!/usr/bin/env python3
"""
SignBridge — Training Script
============================
Extracts hand keypoints from the ASL Alphabet dataset, trains a static letter
classifier (Random Forest + MLP) and a motion classifier (DTW templates for
J and Z).

Run:
    python app/train.py
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import os
import sys
import pickle
import urllib.request
import numpy as np
import pandas as pd
import joblib
import cv2
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Dataset path (Kaggle ASL Alphabet)
RAW_DIR = DATA_DIR / "raw" / "asl_alphabet_train" / "asl_alphabet_train"

# MediaPipe task file
TASK_FILE = MODELS_DIR / "hand_landmarker.task"
TASK_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# CSV output paths
TRAIN_CSV = DATA_DIR / "keypoints_train.csv"
VAL_CSV   = DATA_DIR / "keypoints_val.csv"
TEST_CSV  = DATA_DIR / "keypoints_test.csv"

# Letter sets
# 26 static classes: A-Y excluding J, plus del and space  (J and Z are motion)
# NOTE: 'nothing' is NOT trained — it is handled as a low-confidence fallback at runtime
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["del", "space"]
# All classes extracted from dataset (26 letters + del, space, nothing)
# 'nothing' images are extracted and discarded at load time (no hand landmarks)
ALL_LETTERS    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space", "nothing"]

NUM_LANDMARKS = 21
NUM_FEATURES  = NUM_LANDMARKS * 3   # 63 floats per sample

# MLP hyperparameters
MLP_EPOCHS = 100
MLP_LR     = 1e-3
MLP_BATCH  = 64

# ============================================================
# SECTION 3: DIRECTORY SETUP
# ============================================================
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SECTION 4: DOWNLOAD MEDIAPIPE TASK FILE
# ============================================================
def ensure_task_file():
    """Download hand_landmarker.task if it doesn't already exist."""
    if TASK_FILE.exists():
        return
    print("[↓] Downloading hand_landmarker.task ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)
    print(f"[✓] Saved → {TASK_FILE}")

# ============================================================
# SECTION 5: LANDMARK NORMALIZATION
# ============================================================
def normalize_landmarks(landmarks):
    """
    Normalise a list of 21 NormalizedLandmark objects to a 63-dim vector.
      1. Translate so wrist (landmark 0) is the origin.
      2. Scale by the max absolute value across all coordinates.
      3. Flatten: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]                    # translate wrist to origin
    mx = np.max(np.abs(coords))
    if mx > 0:
        coords /= mx                       # scale to [-1, 1]
    return coords.flatten()

# ============================================================
# SECTION 6: KEYPOINT EXTRACTION
# ============================================================
def _csvs_have_all_classes() -> bool:
    """Return True only if the existing train CSV contains every class in STATIC_LETTERS."""
    try:
        existing = set(pd.read_csv(TRAIN_CSV, usecols=["label"])["label"].unique())
        required = set(STATIC_LETTERS)
        missing  = required - existing
        if missing:
            print(f"  [!] CSVs are missing classes: {sorted(missing)} — re-extracting.")
            return False
        return True
    except Exception:
        return False


def extract_keypoints():
    """
    Run MediaPipe HandLandmarker over every image in the dataset.
    Saves keypoints_train/val/test.csv with an 80/10/10 stratified split.
    Skips only if all three CSVs exist AND contain all expected classes.
    """
    if TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists():
        if _csvs_have_all_classes():
            print("[✓] CSVs already exist and are up to date — skipping extraction.")
            return
        # CSVs are stale — delete and re-extract
        for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
            p.unlink(missing_ok=True)

    if not RAW_DIR.exists():
        print(f"[✗] Dataset not found: {RAW_DIR}")
        print("    Download 'ASL Alphabet' from Kaggle and extract so that")
        print("    data/raw/asl_alphabet_train/asl_alphabet_train/<A-Z>/ exists.")
        sys.exit(1)

    ensure_task_file()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK_FILE)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    all_features, all_labels = [], []

    for letter in ALL_LETTERS:
        letter_dir = RAW_DIR / letter
        if not letter_dir.exists():
            print(f"  [!] Folder missing, skipping: {letter_dir}")
            continue

        paths = sorted(letter_dir.glob("*.jpg")) + sorted(letter_dir.glob("*.png"))
        print(f"  {letter}: {len(paths)} images", flush=True)
        detected = 0

        for p in tqdm(paths, desc=f"    {letter}", leave=False, ncols=70):
            img = cv2.imread(str(p))
            if img is None:
                continue
            rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res    = landmarker.detect(mp_img)
            if res.hand_landmarks:
                all_features.append(normalize_landmarks(res.hand_landmarks[0]))
                all_labels.append(letter)
                detected += 1

        print(f"    → {detected}/{len(paths)} detected")

    landmarker.close()

    if len(all_features) == 0:
        print("[✗] No hands detected in any image. Check dataset path.")
        sys.exit(1)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels)
    print(f"\n[→] Total extracted: {len(X)} samples | {len(set(y))} classes")

    # 80 / 10 / 10 stratified split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    feat_cols = [f"f{i}" for i in range(NUM_FEATURES)]
    for arr_X, arr_y, path in [
        (X_tr, y_tr, TRAIN_CSV),
        (X_va, y_va, VAL_CSV),
        (X_te, y_te, TEST_CSV),
    ]:
        df = pd.DataFrame(arr_X, columns=feat_cols)
        df["label"] = arr_y
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):>6} rows → {path}")

    print("[✓] Extraction complete.\n")

# ============================================================
# SECTION 7: DATA LOADING (static letters only)
# ============================================================
def load_static_data():
    """Load train/val/test splits filtered to STATIC_LETTERS."""
    feat_cols = [f"f{i}" for i in range(NUM_FEATURES)]
    splits = {}
    for name, path in [("train", TRAIN_CSV), ("val", VAL_CSV), ("test", TEST_CSV)]:
        df   = pd.read_csv(path)
        mask = df["label"].isin(STATIC_LETTERS)
        X    = df.loc[mask, feat_cols].values.astype(np.float32)
        y    = df.loc[mask, "label"].values
        splits[name] = (X, y)
        print(f"  {name:5s}: {len(X):>6} samples")
    return splits

# ============================================================
# SECTION 8: MLP ARCHITECTURE
# ============================================================
class StaticMLP(nn.Module):
    """
    Fully-connected classifier for static ASL classes (letters + del/space).
    Input: 63-dim normalised landmark vector.
    Architecture: 256 → 128 → 64 → num_classes
    """
    def __init__(self, input_dim: int = 63, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ============================================================
# SECTION 9: TRAIN RANDOM FOREST
# ============================================================
def train_random_forest(splits):
    print("[→] Training Random Forest (n_estimators=300, n_jobs=-1) ...")
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    va_acc = accuracy_score(y_va, rf.predict(X_va))
    te_acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"  RF   val={va_acc:.4f}  test={te_acc:.4f}")
    return rf, va_acc

# ============================================================
# SECTION 10: TRAIN MLP
# ============================================================
def _augment_batch(Xb: torch.Tensor) -> torch.Tensor:
    """
    Augment a batch of 63-dim landmark vectors at training time.
      - Random Z-axis rotation \u00b120\u00b0 (in-plane hand rotation)
      - Random scale jitter \u00b18 %
      - Gaussian coordinate noise
    Reshapes [B, 63] \u2192 [B, 21, 3] \u2192 transform \u2192 [B, 63].
    """
    B   = Xb.shape[0]
    pts = Xb.view(B, 21, 3).clone()
    angles = (torch.rand(B) - 0.5) * (2 * 3.14159265 * 20 / 360)  # \u00b120 deg
    cos_a  = angles.cos().view(B, 1, 1)
    sin_a  = angles.sin().view(B, 1, 1)
    x, y, z = pts[:, :, 0:1], pts[:, :, 1:2], pts[:, :, 2:3]
    pts = torch.cat([cos_a * x - sin_a * y,
                     sin_a * x + cos_a * y, z], dim=2)
    scale = 1.0 + (torch.rand(B, 1, 1) - 0.5) * 0.16
    pts   = pts * scale + torch.randn_like(pts) * 0.008
    return pts.view(B, 63)


def train_mlp(splits, le: LabelEncoder):
    print("[→] Training MLP ...")
    X_tr, y_tr_raw = splits["train"]
    X_va, y_va_raw = splits["val"]
    X_te, y_te_raw = splits["test"]

    y_tr = le.transform(y_tr_raw)
    y_va = le.transform(y_va_raw)
    y_te = le.transform(y_te_raw)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr),
            torch.tensor(y_tr, dtype=torch.long),
        ),
        batch_size=MLP_BATCH,
        shuffle=True,
    )

    model     = StaticMLP(input_dim=NUM_FEATURES, num_classes=len(le.classes_))
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MLP_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    X_va_t = torch.tensor(X_va)
    X_te_t = torch.tensor(X_te)

    best_va    = 0.0
    best_state = None

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        for Xb, yb in loader:
            Xb = _augment_batch(Xb)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_preds = model(X_va_t).argmax(dim=1).numpy()
        va_acc = accuracy_score(y_va, va_preds)
        scheduler.step()

        if va_acc > best_va:
            best_va    = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MLP_EPOCHS} | val={va_acc:.4f} | best={best_va:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_preds = model(X_te_t).argmax(dim=1).numpy()
    te_acc = accuracy_score(y_te, te_preds)
    print(f"  MLP  val={best_va:.4f}  test={te_acc:.4f}")
    return model, best_va

# ============================================================
# SECTION 11: TRAIN & SAVE BEST STATIC CLASSIFIER
# ============================================================
def train_static_classifier():
    print("\n── Static Classifier ─────────────────────────────────────")
    print("[→] Loading CSV data ...")
    splits = load_static_data()

    le = LabelEncoder()
    le.fit(STATIC_LETTERS)   # consistent ordering: A B C D E F G H I K … Y

    rf,  rf_va  = train_random_forest(splits)
    mlp, mlp_va = train_mlp(splits, le)

    use_mlp = mlp_va >= rf_va
    print(f"\n  Winner: {'MLP' if use_mlp else 'RF'} "
          f"(MLP={mlp_va:.4f}, RF={rf_va:.4f})")

    if use_mlp:
        path = MODELS_DIR / "static_model.pth"
        torch.save({
            "state_dict":  mlp.state_dict(),
            "label_encoder": le,
            "input_dim":   NUM_FEATURES,
            "num_classes": len(le.classes_),
        }, path)
        print(f"[✓] MLP saved → {path}")
    else:
        path = MODELS_DIR / "static_model.pkl"
        joblib.dump({"model": rf, "label_encoder": le}, path)
        print(f"[✓] RF saved → {path}")

    # RF always saved as fallback
    joblib.dump(
        {"model": rf, "label_encoder": le},
        MODELS_DIR / "static_model_rf.pkl",
    )

    # Print per-class report on test set
    X_te, y_te = splits["test"]
    if use_mlp:
        mlp.eval()
        with torch.no_grad():
            preds = le.inverse_transform(
                mlp(torch.tensor(X_te)).argmax(dim=1).numpy()
            )
    else:
        preds = rf.predict(X_te)

    print("\n── Classification Report (Test Set) ──────────────────────")
    print(classification_report(y_te, preds, zero_division=0))

# ============================================================
# SECTION 12: BUILD MOTION CLASSIFIER (DTW Templates)
# ============================================================
def build_motion_model():
    """
    Define reference trajectories for J and Z as numpy arrays of (x, y)
    positions of the index fingertip (landmark 8) relative to the wrist
    (landmark 0), normalised to approximately [-1, 1].

    These templates are matched at runtime using Dynamic Time Warping.
    """
    print("\n── Motion Classifier ─────────────────────────────────────")
    print("[→] Building DTW templates for J and Z ...")

    # J: right-hand index traces a reversed-J
    # Starts high → sweeps straight down → hooks left at the bottom
    J_template = np.array([
        [ 0.22, -0.82],
        [ 0.22, -0.68],
        [ 0.22, -0.54],
        [ 0.22, -0.40],
        [ 0.22, -0.26],
        [ 0.22, -0.12],
        [ 0.22,  0.02],
        [ 0.18,  0.15],
        [ 0.10,  0.26],
        [ 0.00,  0.32],
        [-0.10,  0.30],
        [-0.18,  0.20],
    ], dtype=np.float32)

    # Z: index finger draws a Z
    # Top-left → top-right → diagonal down-left → bottom-right
    Z_template = np.array([
        [-0.40, -0.35],
        [-0.20, -0.35],
        [ 0.00, -0.35],
        [ 0.20, -0.35],
        [ 0.40, -0.35],   # → top-right corner
        [ 0.20, -0.10],
        [ 0.00,  0.05],
        [-0.20,  0.20],
        [-0.40,  0.35],   # → bottom-left corner
        [-0.20,  0.35],
        [ 0.00,  0.35],
        [ 0.20,  0.35],
        [ 0.40,  0.35],   # → bottom-right corner
    ], dtype=np.float32)

    templates = {"J": J_template, "Z": Z_template}

    path = MODELS_DIR / "motion_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(templates, f)
    print(f"[✓] Motion templates saved → {path}")
    return templates

# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 58)
    print("  SignBridge — Training Pipeline")
    print("=" * 58)

    print("\n── Step 1: Feature Extraction ────────────────────────────")
    extract_keypoints()

    train_static_classifier()

    build_motion_model()

    print("\n" + "=" * 58)
    print("  Training complete!  Models saved to models/")
    print("  Next step:  python app/app.py")
    print("=" * 58)
