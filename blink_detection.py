from __future__ import annotations

from typing import List, Optional

import cv2


class BlinkDetector:
    """Approximate blink-rate estimator based on eye detection.

    This helper is designed to be used from the background monitoring thread.
    It relies on OpenCV's Haar-based eye detector and keeps lightweight
    state so it can report a smoothed blink rate in blinks per minute and
    indicate when the rate is below a configurable healthy threshold.
    """

    def __init__(
        self,
        min_blinks_per_minute: float = 12.0,
        alert_cooldown_seconds: float = 600.0,
    ) -> None:
        # Try a more robust cascade first, then fall back to the basic one.
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        if self.eye_cascade.empty():
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )

        self.min_blinks_per_minute = float(min_blinks_per_minute)
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)

        self.last_alert_time: float = 0.0
        self.last_eye_state_open: Optional[bool] = None
        self.eye_closed_start: Optional[float] = None

        # Recent blinks within a sliding time window (for live rate)
        self.recent_blinks: List[float] = []
        # All blinks seen in the current monitoring session (for summary stats)
        self.all_blinks: List[float] = []

    def reset(self) -> None:
        self.last_alert_time = 0.0
        self.last_eye_state_open = None
        self.eye_closed_start = None
        self.recent_blinks.clear()
        self.all_blinks.clear()

    def update(self, gray_frame, face_rect, now: float) -> dict:
        """Update blink state from a new frame.

        Parameters
        ----------
        gray_frame: numpy.ndarray
            Full grayscale frame from the webcam.
        face_rect: tuple[int, int, int, int]
            (x, y, w, h) rectangle of the detected face in the frame.
        now: float
            Current time.time() value.

        Returns
        -------
        dict with keys:
            - blink_rate_bpm: float
            - low_blink_alert: bool
        """

        if self.eye_cascade.empty():
            return {}

        x, y, w, h = face_rect
        roi = gray_frame[y : y + h, x : x + w]
        if roi.size == 0:
            return {}

        eyes = self.eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
        )

        eyes_open = len(eyes) > 0
        prev_open = self.last_eye_state_open
        self.last_eye_state_open = eyes_open

        # Detect short closed-then-open sequences as blinks.
        if prev_open is True and not eyes_open:
            # Potential start of a blink.
            self.eye_closed_start = now
        elif prev_open is False and eyes_open:
            # Eyes have just opened again.
            if self.eye_closed_start is not None:
                closed_duration = now - self.eye_closed_start
                # Ignore very long closures or ultra-short noise.
                if 0.05 < closed_duration < 0.8:
                    self.recent_blinks.append(now)
                    self.all_blinks.append(now)
                self.eye_closed_start = None

        # Maintain a sliding window of recent blinks (last ~60s).
        window_seconds = 60.0
        self.recent_blinks = [t for t in self.recent_blinks if now - t <= window_seconds]

        blink_rate_bpm = 0.0
        if self.recent_blinks:
            window_span = max(now - self.recent_blinks[0], 1.0)
            blink_rate_bpm = len(self.recent_blinks) * 60.0 / window_span

        info = {
            "blink_rate_bpm": blink_rate_bpm,
            "low_blink_alert": False,
        }

        # Only evaluate low-blink alerts after at least ~30s of observation.
        observation_time = now - self.recent_blinks[0] if self.recent_blinks else 0.0
        if observation_time >= 30.0 and blink_rate_bpm > 0.0:
            if blink_rate_bpm < self.min_blinks_per_minute:
                if now - self.last_alert_time >= self.alert_cooldown_seconds:
                    self.last_alert_time = now
                    info["low_blink_alert"] = True

        return info

    def session_stats(self, session_start: float, session_end: float) -> dict:
        """Return aggregate blink stats for a completed monitoring session."""

        if not self.all_blinks:
            return {
                "avg_blink_rate_per_minute": 0.0,
                "total_blinks": 0,
            }

        duration_min = max((session_end - session_start) / 60.0, 1e-3)
        total_blinks = len(self.all_blinks)
        avg_rate = total_blinks / duration_min

        return {
            "avg_blink_rate_per_minute": round(avg_rate, 2),
            "total_blinks": int(total_blinks),
        }
