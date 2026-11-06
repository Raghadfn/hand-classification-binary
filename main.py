"""
Real-time Hand Gesture Number Classification
==============================================
Classifies hand gestures into digits 0-6 using a pre-trained Keras model,
MediaPipe hand detection, temporal voting for stability, and CustomTkinter GUI.

Model: D:/pro/New folder2/model.keras
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = r"D:/pro/New folder2/model.keras"
CAMERA_INDEX = 0
INPUT_SIZE = (64, 64)
ROI_PADDING = 0.35
BACKGROUND_GRAY = 50
MIRROR_FRAME = True

VOTE_WINDOW_SIZE = 5
MIN_VOTE_CONFIDENCE = 0.55
MIN_DISPLAY_CONFIDENCE = 0.55

UI_POLL_INTERVAL_MS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FrameResult:
    """Payload sent from the inference thread to the GUI thread."""
    frame_bgr: np.ndarray
    processed_roi: Optional[np.ndarray]
    raw_prediction: Optional[int]
    raw_confidence: float
    stable_prediction: Optional[int]
    stable_confidence: float
    fps: float
    hand_detected: bool
    is_running: bool = True


@dataclass
class PredictionVoter:
    """Majority-vote filter over the last N confident predictions."""
    window_size: int = VOTE_WINDOW_SIZE
    min_confidence: float = MIN_VOTE_CONFIDENCE
    _history: Deque[Tuple[int, float]] = field(default_factory=collections.deque)

    def __post_init__(self) -> None:
        self._history = collections.deque(maxlen=self.window_size)

    def add(self, prediction: Optional[int], confidence: float) -> None:
        if prediction is None or confidence < self.min_confidence:
            return
        self._history.append((prediction, confidence))

    def get_stable(self) -> Tuple[Optional[int], float]:
        if not self._history:
            return None, 0.0

        votes: dict[int, list[float]] = {}
        for cls, conf in self._history:
            votes.setdefault(cls, []).append(conf)

        winner = max(votes, key=lambda c: len(votes[c]))
        avg_conf = float(np.mean(votes[winner]))
        return winner, avg_conf

    def reset(self) -> None:
        self._history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
class HandPreprocessor:
    """Converts a raw BGR hand crop into a model-ready tensor."""

    @staticmethod
    def square_bbox_from_landmarks(
        landmarks,
        frame_w: int,
        frame_h: int,
        padding: float = ROI_PADDING,
    ) -> Optional[Tuple[int, int, int, int]]:
        xs = [lm.x * frame_w for lm in landmarks.landmark]
        ys = [lm.y * frame_h for lm in landmarks.landmark]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        side = max(x_max - x_min, y_max - y_min) * (1.0 + padding)
        half = side / 2.0

        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(frame_w, cx + half))
        y2 = int(min(frame_h, cy + half))

        w, h = x2 - x1, y2 - y1
        if w <= 20 or h <= 20:
            return None

        if w > h:
            diff = w - h
            y1 = max(0, y1 - diff // 2)
            y2 = min(frame_h, y2 + (diff - diff // 2))
        elif h > w:
            diff = h - w
            x1 = max(0, x1 - diff // 2)
            x2 = min(frame_w, x2 + (diff - diff // 2))

        return x1, y1, x2, y2

    @staticmethod
    def to_model_input(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        background = np.full(gray.shape, BACKGROUND_GRAY, dtype=np.uint8)
        final = np.where(mask == 255, gray, background).astype(np.uint8)
        final = cv2.bilateralFilter(final, 5, 50, 50)

        preview = cv2.resize(final, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        normalized = preview.astype(np.float32) / 255.0
        model_input = np.expand_dims(normalized, axis=(0, -1))
        return model_input, preview


# ─────────────────────────────────────────────────────────────────────────────
# Background inference thread
# ─────────────────────────────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    """Capture -> detect -> preprocess -> infer loop on a background thread."""

    def __init__(
        self,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        pause_event: threading.Event,
        model: tf.keras.Model,
    ) -> None:
        super().__init__(daemon=True, name="InferenceWorker")
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.model = model
        self.voter = PredictionVoter()

    def run(self) -> None:
        cap = self._open_camera()
        if cap is None:
            self._emit_error_frame("Camera not available")
            return

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        prev_time = time.perf_counter()
        fps = 0.0

        try:
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok:
                    log.warning("Failed to read frame from camera")
                    time.sleep(0.05)
                    continue

                if MIRROR_FRAME:
                    frame = cv2.flip(frame, 1)

                display = frame.copy()
                h, w = frame.shape[:2]

                raw_pred: Optional[int] = None
                raw_conf = 0.0
                processed_roi = None
                hand_detected = False
                stable_pred: Optional[int] = None
                stable_conf = 0.0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = hands.process(rgb)

                if mp_result.multi_hand_landmarks:
                    hand_detected = True
                    lms = mp_result.multi_hand_landmarks[0]
                    bbox = HandPreprocessor.square_bbox_from_landmarks(lms, w, h)

                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        roi = frame[y1:y2, x1:x2]

                        model_input, processed_roi = HandPreprocessor.to_model_input(roi)
                        probs = self.model(model_input, training=False).numpy()[0]
                        raw_pred = int(np.argmax(probs))
                        raw_conf = float(probs[raw_pred])

                        self.voter.add(raw_pred, raw_conf)
                        stable_pred, stable_conf = self.voter.get_stable()

                        box_color = (
                            (0, 230, 160) if raw_conf >= MIN_DISPLAY_CONFIDENCE
                            else (0, 160, 255)
                        )
                        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                        label = f"{raw_pred}  {raw_conf * 100:.0f}%"
                        cv2.putText(
                            display, label, (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2,
                        )
                        mp.solutions.drawing_utils.draw_landmarks(
                            display, lms, mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style(),
                        )
                    else:
                        stable_pred, stable_conf = self.voter.get_stable()
                else:
                    stable_pred, stable_conf = self.voter.get_stable()

                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
                prev_time = now

                payload = FrameResult(
                    frame_bgr=display,
                    processed_roi=processed_roi,
                    raw_prediction=raw_pred,
                    raw_confidence=raw_conf,
                    stable_prediction=stable_pred if hand_detected else None,
                    stable_confidence=stable_conf if hand_detected else 0.0,
                    fps=fps,
                    hand_detected=hand_detected,
                )
                self._push(payload)

        finally:
            hands.close()
            cap.release()
            log.info("Inference worker stopped")

    @staticmethod
    def _open_camera() -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            log.error("Cannot open camera index %d", CAMERA_INDEX)
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        log.info("Camera opened successfully")
        return cap

    def _push(self, payload: FrameResult) -> None:
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.result_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _emit_error_frame(self, message: str) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank, message, (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2,
        )
        self._push(FrameResult(
            frame_bgr=blank,
            processed_roi=None,
            raw_prediction=None,
            raw_confidence=0.0,
            stable_prediction=None,
            stable_confidence=0.0,
            fps=0.0,
            hand_detected=False,
            is_running=False,
        ))

    def reset_voter(self) -> None:
        self.voter.reset()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class GestureApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Hand Gesture Number Recognition")
        self.geometry("1020x740")
        self.minsize(960, 680)

        self.result_queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.worker: Optional[InferenceWorker] = None
        self.model: Optional[tf.keras.Model] = None

        self._video_photo: Optional[ImageTk.PhotoImage] = None
        self._roi_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._load_model()
        self.after(UI_POLL_INTERVAL_MS, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        container = ctk.CTkFrame(self, corner_radius=16)
        container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        container.grid_columnconfigure(0, weight=3)
        container.grid_columnconfigure(1, weight=2)
        container.grid_rowconfigure(0, weight=1)

        video_panel = ctk.CTkFrame(container, corner_radius=12)
        video_panel.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        video_panel.grid_rowconfigure(1, weight=1)
        video_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            video_panel, text="Live Camera Feed",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")

        self.video_label = ctk.CTkLabel(video_panel, text="Press Start to begin")
        self.video_label.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        side = ctk.CTkFrame(container, corner_radius=12)
        side.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        side.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            side, text="Prediction",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=24, pady=(24, 4), sticky="w")

        self.digit_label = ctk.CTkLabel(
            side, text="—",
            font=ctk.CTkFont(size=110, weight="bold"),
            text_color="#00E5A8",
        )
        self.digit_label.grid(row=1, column=0, padx=24, pady=4)

        self.conf_label = ctk.CTkLabel(
            side, text="Confidence: —",
            font=ctk.CTkFont(size=20),
        )
        self.conf_label.grid(row=2, column=0, padx=24, pady=(8, 2), sticky="w")

        self.fps_label = ctk.CTkLabel(
            side, text="FPS: —",
            font=ctk.CTkFont(size=17),
            text_color="#888888",
        )
        self.fps_label.grid(row=3, column=0, padx=24, pady=(2, 12), sticky="w")

        self.status_label = ctk.CTkLabel(
            side, text="Ready",
            font=ctk.CTkFont(size=14),
            wraplength=280, justify="left",
            text_color="#AAAAAA",
        )
        self.status_label.grid(row=4, column=0, padx=24, pady=(0, 12), sticky="w")

        ctk.CTkLabel(
            side, text="Model Input (64 x 64)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=5, column=0, padx=24, pady=(4, 4), sticky="w")

        self.roi_label = ctk.CTkLabel(side, text="")
        self.roi_label.grid(row=6, column=0, padx=24, pady=(0, 16))

        btn_frame = ctk.CTkFrame(side, fg_color="transparent")
        btn_frame.grid(row=7, column=0, padx=24, pady=(0, 24), sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="Start", command=self._start_stream,
            fg_color="#00A86B", hover_color="#008F5A",
            font=ctk.CTkFont(size=15, weight="bold"), height=40,
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="Stop", command=self._stop_stream,
            fg_color="#C0392B", hover_color="#96281B",
            font=ctk.CTkFont(size=15, weight="bold"), height=40,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

    def _load_model(self) -> None:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            msg = f"Model not found:\n{MODEL_PATH}"
            log.error(msg)
            self.status_label.configure(text=msg, text_color="#FF6B6B")
            self.start_btn.configure(state="disabled")
            return

        try:
            self.model = tf.keras.models.load_model(str(model_path))
            in_shape = self.model.input_shape
            out_shape = self.model.output_shape
            log.info("Model loaded - input: %s, output: %s", in_shape, out_shape)
            self.status_label.configure(
                text=f"Model ready | input {in_shape} -> {out_shape[-1]} classes",
                text_color="#AAAAAA",
            )
        except Exception as exc:
            log.exception("Failed to load model")
            self.status_label.configure(
                text=f"Model load error:\n{exc}", text_color="#FF6B6B"
            )
            self.start_btn.configure(state="disabled")

    def _start_stream(self) -> None:
        if self.model is None:
            return

        if self.worker is None or not self.worker.is_alive():
            self.worker = InferenceWorker(
                self.result_queue, self.stop_event, self.pause_event, self.model
            )
            self.worker.start()

        self.pause_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Stream running...", text_color="#00E5A8")
        log.info("Stream started")

    def _stop_stream(self) -> None:
        self.pause_event.set()
        if self.worker:
            self.worker.reset_voter()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.digit_label.configure(text="—", text_color="#555555")
        self.conf_label.configure(text="Confidence: —")
        self.fps_label.configure(text="FPS: —")
        self.status_label.configure(text="Stream stopped", text_color="#AAAAAA")
        log.info("Stream stopped")

    def _poll(self) -> None:
        try:
            result = self.result_queue.get_nowait()
            self._render(result)
        except queue.Empty:
            pass
        self.after(UI_POLL_INTERVAL_MS, self._poll)

    def _render(self, r: FrameResult) -> None:
        rgb = cv2.cvtColor(r.frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((660, 500), Image.Resampling.LANCZOS)
        self._video_photo = ImageTk.PhotoImage(pil)
        self.video_label.configure(image=self._video_photo, text="")

        if r.processed_roi is not None:
            roi_rgb = cv2.cvtColor(r.processed_roi, cv2.COLOR_GRAY2RGB)
            roi_pil = Image.fromarray(roi_rgb).resize((120, 120), Image.Resampling.NEAREST)
            self._roi_photo = ImageTk.PhotoImage(roi_pil)
            self.roi_label.configure(image=self._roi_photo)

        if not self.pause_event.is_set():
            self.fps_label.configure(text=f"FPS: {r.fps:.1f}")

        if not r.hand_detected:
            self.digit_label.configure(text="—", text_color="#555555")
            self.conf_label.configure(text="Confidence: —")
            self.status_label.configure(text="No hand detected - show your hand")
            return

        pred = r.stable_prediction
        conf = r.stable_confidence

        if pred is None or conf < MIN_DISPLAY_CONFIDENCE:
            self.digit_label.configure(text="?", text_color="#FFB347")
            self.conf_label.configure(text=f"Confidence: {r.raw_confidence * 100:.1f}%")
            self.status_label.configure(text="Hand detected - hold gesture steady")
            return

        color = "#00E5A8" if conf >= 0.75 else "#FFB347"
        self.digit_label.configure(text=str(pred), text_color=color)
        self.conf_label.configure(text=f"Confidence: {conf * 100:.1f}%")
        self.status_label.configure(text="Stable prediction", text_color="#00E5A8")

    def _on_close(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.destroy()
        log.info("Application closed")


def main() -> None:
    app = GestureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
