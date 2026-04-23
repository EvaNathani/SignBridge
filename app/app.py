#!/usr/bin/env python3
"""
SignBridge — Live ASL Fingerspelling Recognition
=================================================
Loads trained models and runs a real-time recognition loop via webcam.

Controls:
  SPACE     — add space to sentence buffer
  BACKSPACE — delete last character
  ENTER     — clear sentence buffer
  Q         — quit

Run:
    python app/app.py
"""

# ============================================================
# SECTION 1: IMPORTS
# ============================================================
import sys
import time
import pickle
import urllib.request
import numpy as np
import joblib
import cv2
from pathlib import Path
from collections import deque, Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import torch
import torch.nn as nn

# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

TASK_FILE = MODELS_DIR / "hand_landmarker.task"
TASK_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

NUM_LANDMARKS = 21
NUM_FEATURES  = NUM_LANDMARKS * 3   # 63

# 26 static classes (must match train.py)
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["del", "space"]

# ── Hold / stability logic ──────────────────────────────────
HOLD_FRAMES     = 45    # consecutive stable frames before confirming (~1.5s @ 30fps)
COOLDOWN_FRAMES = 40    # frames to ignore after a confirmation
VOTE_WINDOW     = 11    # majority-vote window size

# ── Motion detection ────────────────────────────────────────
MOTION_BUFFER_SIZE    = 20    # number of frames in rolling trajectory buffer
MOTION_MIN_DISP       = 0.06  # minimum normalised displacement to classify motion
MOTION_CONF_THRESHOLD = 0.45  # DTW confidence threshold

# ── Display ─────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (0, 230,   0)
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GRAY   = (160, 160, 160)
YELLOW = (  0, 220, 255)
ORANGE = (  0, 140, 255)

# Hand skeleton connections (MediaPipe landmark indices)
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index
    (5, 9),  (9, 10), (10,11), (11,12),   # middle
    (9, 13), (13,14), (14,15), (15,16),   # ring
    (13,17), (17,18), (18,19), (19,20),   # pinky
    (0, 17),                               # palm base
]

# ============================================================
# SECTION 3: MODEL ARCHITECTURE (must match train.py exactly)
# ============================================================
class StaticMLP(nn.Module):
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
# SECTION 4: MODEL LOADING
# ============================================================
def load_static_model():
    """Load the best static classifier (MLP preferred, RF fallback)."""
    mlp_path = MODELS_DIR / "static_model.pth"
    rf_path  = MODELS_DIR / "static_model.pkl"

    if mlp_path.exists():
        ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)
        model = StaticMLP(input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[✓] Static model: MLP ({ckpt['num_classes']} classes)")
        return model, ckpt["label_encoder"], "mlp"

    if rf_path.exists():
        data = joblib.load(rf_path)
        print("[✓] Static model: Random Forest")
        return data["model"], data["label_encoder"], "rf"

    print("[✗] No static model found!  Run:  python app/train.py")
    sys.exit(1)


def load_motion_model():
    """Load DTW template trajectories for J and Z."""
    path = MODELS_DIR / "motion_model.pkl"
    if not path.exists():
        print("[✗] No motion model found!  Run:  python app/train.py")
        sys.exit(1)
    with open(path, "rb") as f:
        templates = pickle.load(f)
    print(f"[✓] Motion model: DTW templates for {list(templates.keys())}")
    return templates

# ============================================================
# SECTION 5: MEDIAPIPE SETUP
# ============================================================
def ensure_task_file():
    """Download hand_landmarker.task if missing."""
    if TASK_FILE.exists():
        return
    print("[↓] Downloading hand_landmarker.task ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)
    print(f"[✓] Downloaded → {TASK_FILE}")


def setup_landmarker():
    """Create a HandLandmarker configured for VIDEO (sequential frame) mode."""
    ensure_task_file()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK_FILE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

# ============================================================
# SECTION 6: FEATURE EXTRACTION HELPERS
# ============================================================
def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Translate wrist to origin, scale by max-abs, flatten to 63-dim float32.
    Identical to the function used during training.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]
    mx = np.max(np.abs(coords))
    if mx > 0:
        coords /= mx
    return coords.flatten()


def get_index_tip_xy(landmarks) -> np.ndarray:
    """Return (x, y) of index fingertip (lm 8) relative to wrist (lm 0)."""
    return np.array(
        [landmarks[8].x - landmarks[0].x,
         landmarks[8].y - landmarks[0].y],
        dtype=np.float32,
    )

# ============================================================
# SECTION 7: STATIC INFERENCE
# ============================================================
def predict_static(model, le, model_type: str, features: np.ndarray):
    """Return (predicted_letter, confidence) from the static classifier."""
    if model_type == "mlp":
        with torch.no_grad():
            logits = model(torch.tensor(features).unsqueeze(0))
            probs  = torch.softmax(logits, dim=1).numpy()[0]
    else:
        probs = model.predict_proba([features])[0]

    idx    = int(np.argmax(probs))
    letter = le.inverse_transform([idx])[0]
    return letter, float(probs[idx])

# ============================================================
# SECTION 8: MOTION INFERENCE (DTW)
# ============================================================
def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """
    Simple O(n·m) Dynamic Time Warping distance between two 2-D sequences.
    Each row is an (x, y) point; distance is Euclidean.
    """
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[n, m])


def classify_motion(traj_list: list, templates: dict):
    """
    Match a rolling trajectory of index-tip positions against J / Z templates.

    Returns (letter, confidence) if a good match is found, else (None, 0.0).
    The trajectory is normalised relative to its start position and span before
    DTW so that scale and absolute position are irrelevant.
    """
    if len(traj_list) < 8:
        return None, 0.0

    traj = np.array(traj_list, dtype=np.float32)
    traj -= traj[0]                                 # relative to trajectory start
    disp  = float(np.max(np.abs(traj)))
    if disp < MOTION_MIN_DISP:
        return None, 0.0                            # hand is essentially static
    traj /= disp                                    # normalise scale

    best_letter, best_dist = None, np.inf
    for letter, template in templates.items():
        d = _dtw(traj, template)
        if d < best_dist:
            best_dist   = d
            best_letter = letter

    # Map distance to [0, 1] confidence (distance < 2 → ~1.0; > 10 → ~0)
    confidence = max(0.0, min(1.0, 1.0 - (best_dist - 2.0) / 8.0))

    if confidence >= MOTION_CONF_THRESHOLD:
        return best_letter, confidence
    return None, 0.0

# ============================================================
# SECTION 9: DRAWING HELPERS
# ============================================================
def draw_skeleton(frame: np.ndarray, hand_landmarks_list: list):
    """Render all detected hand skeletons onto the frame in-place."""
    h, w = frame.shape[:2]
    for landmarks in hand_landmarks_list:
        # Connections
        for a, b in HAND_CONNECTIONS:
            ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
            bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), GREEN, 2, cv2.LINE_AA)
        # Joints
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, WHITE, -1)
            cv2.circle(frame, (cx, cy), 5, GREEN, 2)


def draw_ui(
    frame:      np.ndarray,
    letter:     str | None,
    conf:       float,
    sentence:   str,
    hold_frac:  float,
    in_cooldown: bool,
    fps:        float,
):
    """Render all UI elements onto the frame in-place."""
    h, w = frame.shape[:2]

    # ── Top translucent bar ──────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Large predicted letter (top-left)
    disp_letter = letter if letter else "?"
    letter_color = WHITE if letter else GRAY
    cv2.putText(frame, disp_letter, (18, 62), FONT, 2.4, letter_color, 4, cv2.LINE_AA)

    # Confidence and FPS (top-right)
    conf_str = f"{conf:.0%}" if conf > 0 else "--"
    cv2.putText(frame, f"Conf: {conf_str}", (w - 195, 30), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:  {fps:5.1f}",  (w - 195, 58), FONT, 0.65, GRAY,  1, cv2.LINE_AA)

    # ── Hold progress bar ────────────────────────────────────
    bx, by, bw, bh = 18, 80, w - 36, 11
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)

    if in_cooldown:
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), ORANGE, -1)
        cd_text = "COOLDOWN"
        tw = cv2.getTextSize(cd_text, FONT, 0.42, 1)[0][0]
        cv2.putText(frame, cd_text, (bx + bw // 2 - tw // 2, by + 9),
                    FONT, 0.42, WHITE, 1, cv2.LINE_AA)
    elif hold_frac > 0:
        fill  = int(bw * hold_frac)
        color = GREEN if hold_frac >= 1.0 else YELLOW
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color, -1)

    # ── Bottom translucent bar ───────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 78), (w, h), BLACK, -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    # Sentence buffer
    disp_sentence = sentence if sentence else "(empty — sign a letter to begin)"
    # Truncate left if too long (~50 chars fits at scale 0.72)
    if len(disp_sentence) > 52:
        disp_sentence = "…" + disp_sentence[-51:]
    cv2.putText(frame, disp_sentence, (12, h - 42), FONT, 0.72, WHITE, 2, cv2.LINE_AA)

    # Legend (bottom-right, small)
    legend = ["SPC: space", "BKSP: delete", "ENTER: clear", "Q: quit"]
    for i, text in enumerate(legend):
        cv2.putText(frame, text,
                    (w - 155, h - 72 + i * 17),
                    FONT, 0.38, GRAY, 1, cv2.LINE_AA)

# ============================================================
# SECTION 10: MAIN APPLICATION LOOP
# ============================================================
def run():
    # ── Load models ─────────────────────────────────────────
    print("[→] Loading models ...")
    static_model, label_encoder, model_type = load_static_model()
    motion_templates = load_motion_model()

    # ── MediaPipe ───────────────────────────────────────────
    print("[→] Initialising MediaPipe HandLandmarker ...")
    landmarker = setup_landmarker()

    # ── Webcam ──────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Cannot open webcam. Check camera connection.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[✓] SignBridge running — press Q to quit.\n")

    # ── App state ────────────────────────────────────────────
    sentence        = ""
    pred_buffer     = deque(maxlen=VOTE_WINDOW)
    motion_buffer   = deque(maxlen=MOTION_BUFFER_SIZE)
    tracked_letter  = None     # letter currently being held
    hold_counter    = 0
    cooldown_counter = 0

    # FPS measurement
    t_start   = time.perf_counter()
    fps_timer = time.perf_counter()
    fps_count = 0
    fps       = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Frame read failed — retrying ...")
            continue

        # Mirror feed so it feels like a mirror
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── MediaPipe inference ──────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.perf_counter() - t_start) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        current_letter = None
        current_conf   = 0.0

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Draw skeleton
            draw_skeleton(frame, result.hand_landmarks)

            # Static prediction
            features      = normalize_landmarks(landmarks)
            static_letter, static_conf = predict_static(
                static_model, label_encoder, model_type, features
            )

            # Update motion trajectory buffer
            motion_buffer.append(get_index_tip_xy(landmarks))

            # Motion prediction
            motion_letter, motion_conf = classify_motion(
                list(motion_buffer), motion_templates
            )

            # Fusion: motion wins when confident
            if motion_letter and motion_conf > 0.5:
                current_letter = motion_letter
                current_conf   = motion_conf
            elif static_conf < 0.35:
                # Hand visible but no confident match — treat as nothing
                current_letter = None
                current_conf   = 0.0
            else:
                current_letter = static_letter
                current_conf   = static_conf
        else:
            # No hand visible — reset all tracking state
            motion_buffer.clear()
            pred_buffer.clear()
            tracked_letter  = None
            hold_counter    = 0

        # ── Hold / cooldown logic ─────────────────────────────
        if cooldown_counter > 0:
            cooldown_counter -= 1
            hold_counter = 0

        elif current_letter:
            pred_buffer.append(current_letter)

            if len(pred_buffer) >= VOTE_WINDOW:
                voted = Counter(pred_buffer).most_common(1)[0][0]

                if voted == tracked_letter:
                    hold_counter += 1
                else:
                    tracked_letter = voted
                    hold_counter   = 1

                current_letter = tracked_letter   # display the stable letter

                if hold_counter >= HOLD_FRAMES:
                    # Act on special classes instead of appending literally
                    if tracked_letter == "space":
                        sentence += " "
                    elif tracked_letter == "del":
                        sentence = sentence[:-1]
                    elif tracked_letter == "nothing":
                        pass   # intentional no-op
                    else:
                        sentence += tracked_letter
                    print(f"  Confirmed: [{tracked_letter}]  buffer: \"{sentence}\"")
                    hold_counter      = 0
                    cooldown_counter  = COOLDOWN_FRAMES
                    tracked_letter    = None
                    pred_buffer.clear()
        else:
            tracked_letter = None
            hold_counter   = 0

        hold_frac = min(hold_counter / HOLD_FRAMES, 1.0) if HOLD_FRAMES > 0 else 0.0

        # ── FPS counter ──────────────────────────────────────
        fps_count += 1
        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            fps       = fps_count / (now - fps_timer)
            fps_count = 0
            fps_timer = now

        # ── Render UI ────────────────────────────────────────
        draw_ui(
            frame,
            current_letter,
            current_conf,
            sentence,
            hold_frac,
            cooldown_counter > 0,
            fps,
        )

        cv2.imshow("SignBridge — Real-time ASL Recognition", frame)

        # ── Keyboard controls + window-close button ───────────
        key = cv2.waitKey(1) & 0xFF
        # Detect the X button being clicked (window property becomes -1)
        window_closed = (
            cv2.getWindowProperty(
                "SignBridge — Real-time ASL Recognition",
                cv2.WND_PROP_VISIBLE,
            ) < 1
        )
        if window_closed or key in (ord('q'), ord('Q')):
            break
        elif key == 32:   # SPACE
            sentence += " "
        elif key == 8:    # BACKSPACE
            sentence = sentence[:-1]
        elif key == 13:   # ENTER
            sentence = ""

    # ── Cleanup ──────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\n[✓] SignBridge closed.")
    if sentence.strip():
        print(f'    Final sentence: "{sentence}"')


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run()
