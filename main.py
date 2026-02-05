import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Union
import sys
import random

import cv2
from plyer import notification
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsBlurEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from stats_dashboard import StatsDashboardWindow
from eye_test_page import EyeTestPage
from blink_detection import BlinkDetector
from posture_detection import PostureDetector


class UsageLogger:
    """Simple JSON logger that keeps all data in a single list in usage_log.json."""

    def __init__(self, filepath: str = "usage_log.json") -> None:
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_all(self):
        self._ensure_file()
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def append_entry(self, entry: dict) -> None:
        data = self._read_all()
        data.append(entry)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class NotificationManager:
    """Desktop notifications using plyer, with a console fallback."""

    def __init__(self, app_name: str = "VisionMate") -> None:
        self.app_name = app_name

    def notify(self, title: str, message: str, timeout: int = 5) -> None:
        try:
            notification.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                timeout=timeout,
            )
        except Exception:
            print(f"[{self.app_name}] {title}: {message}")


class MonitorThread(threading.Thread):
    """Background thread that reads webcam frames and detects face + distance.

    It posts high‑level events into a thread‑safe queue that the GUI consumes.
    """

    def __init__(
        self,
        event_queue: "queue.Queue[tuple]",
        min_distance_cm: float = 40.0,
        break_interval_seconds: float = 2100.0,
        frame_interval: float = 0.3,
    ) -> None:
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.min_distance_cm = float(min_distance_cm)
        # Continuous viewing time (in seconds) before a break reminder is fired.
        self.break_interval_seconds = float(break_interval_seconds)
        self.frame_interval = float(frame_interval)

        self.stop_event = threading.Event()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.reference_face_width_px = 150.0
        self.reference_distance_cm = 60.0

        self.last_frame_time = None
        self.continuous_face_time = 0.0
        self.last_face_seen_time = None
        # Reset continuous viewing timer only after being away from the screen
        # for this many seconds (5 minutes by default).
        self.absence_reset_seconds = 300.0

        self.last_distance_alert_time = 0.0
        # Short cooldown so warnings feel continuous while user is too close
        self.distance_alert_cooldown = 5.0

        self.session_start_time = time.time()
        self.too_close_events = 0
        self.break_alerts = 0

        # Additional monitors for blink rate and posture.
        self.blink_detector = BlinkDetector()
        self.posture_detector = PostureDetector()

    def run(self) -> None:
        if self.face_cascade.empty():
            self._post_event(
                "error",
                {"message": "Could not load face detection model (Haar cascade)."},
            )
            return

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._post_event("error", {"message": "Unable to access the webcam."})
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.last_frame_time = time.time()

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                now = time.time()

                if not ret:
                    time.sleep(self.frame_interval)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80),
                )

                dt = now - self.last_frame_time if self.last_frame_time else 0.0
                self.last_frame_time = now

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    self._update_timers_face_present(now, dt)

                    distance_cm = self._estimate_distance_cm(w)
                    if distance_cm is not None:
                        # Always send live distance updates while a face is detected
                        self._post_event(
                            "distance_update",
                            {"distance_cm": round(distance_cm, 1)},
                        )

                        # Trigger warnings when the user is too close
                        if distance_cm < self.min_distance_cm:
                            if (
                                now - self.last_distance_alert_time
                                >= self.distance_alert_cooldown
                            ):
                                self.too_close_events += 1
                                self.last_distance_alert_time = now
                                self._post_event(
                                    "distance_warning",
                                    {"distance_cm": round(distance_cm, 1)},
                                )
                    else:
                        # Face detected but distance could not be estimated
                        self._post_event("distance_update", {"distance_cm": None})

                    face_rect = (x, y, w, h)

                    # Blink-rate monitoring.
                    if self.blink_detector is not None:
                        blink_info = self.blink_detector.update(gray, face_rect, now)
                        if blink_info:
                            bpm = blink_info.get("blink_rate_bpm")
                            if bpm is not None:
                                self._post_event(
                                    "blink_rate_update",
                                    {"blink_rate_bpm": float(bpm)},
                                )
                            if blink_info.get("low_blink_alert"):
                                self._post_event(
                                    "blink_low",
                                    {"blink_rate_bpm": float(bpm or 0.0)},
                                )

                    # Posture monitoring.
                    if self.posture_detector is not None:
                        posture_info = self.posture_detector.update(face_rect, gray.shape, now)
                        state = posture_info.get("state")
                        if state:
                            self._post_event("posture_update", {"state": state})
                        if posture_info.get("posture_alert"):
                            self._post_event("posture_warning", {"state": state})

                    if self.continuous_face_time >= self.break_interval_seconds:
                        self.break_alerts += 1
                        self.continuous_face_time = 0.0
                        self._post_event("break_reminder", {})
                else:
                    self._update_timers_no_face(now, dt)
                    # No face detected; clear distance in the UI
                    self._post_event("distance_update", {"distance_cm": None})

                time.sleep(self.frame_interval)
        finally:
            cap.release()

    def stop(self) -> None:
        self.stop_event.set()

    def _post_event(self, name: str, payload: dict) -> None:
        try:
            self.event_queue.put((name, payload), block=False)
        except queue.Full:
            pass

    def _update_timers_face_present(self, now: float, dt: float) -> None:
        self.last_face_seen_time = now
        self.continuous_face_time += dt

    def _update_timers_no_face(self, now: float, dt: float) -> None:  # noqa: ARG002
        if self.last_face_seen_time is not None:
            if now - self.last_face_seen_time >= self.absence_reset_seconds:
                self.continuous_face_time = 0.0

    def _estimate_distance_cm(self, face_width_px: float):
        if face_width_px <= 0:
            return None
        return (
            self.reference_distance_cm
            * self.reference_face_width_px
            / float(face_width_px)
        )

    def get_session_stats(self) -> dict:
        end_time = time.time()
        stats: dict[str, object] = {
            "type": "monitoring_session",
            "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(
                timespec="seconds"
            ),
            "end_time": datetime.fromtimestamp(end_time).isoformat(timespec="seconds"),
            "total_monitor_seconds": round(end_time - self.session_start_time),
            "too_close_events": int(self.too_close_events),
            "break_alerts": int(self.break_alerts),
        }

        # Merge in aggregate blink metrics, if available.
        try:
            if self.blink_detector is not None:
                blink_stats = self.blink_detector.session_stats(
                    self.session_start_time,
                    end_time,
                )
                stats.update(blink_stats)
        except Exception:  # noqa: BLE001
            pass

        # Merge in aggregate posture metrics, if available.
        try:
            if self.posture_detector is not None:
                posture_stats = self.posture_detector.session_stats()
                stats.update(posture_stats)
        except Exception:  # noqa: BLE001
            pass

        return stats


class VisionTestDialog(QDialog):
    """Simple, robust Qt-only vision self-test (no external imaging).

    The test shows random letters or short words in different font sizes
    and with different blur levels, using Qt's own rendering. The user
    types what they see, answers are scored, and a Good/Average/Poor
    classification is calculated. Results are stored via UsageLogger.

    This is only a rough self-check and *not* a medical diagnosis.
    """

    def __init__(self, logger: UsageLogger, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vision self-assessment test")
        self.setModal(True)
        self.logger = logger

        # Pre-generate all test cases so they can be scored and logged
        # consistently. Each case represents one item shown to the user.
        self.test_cases = self._build_test_cases()
        # Index of the currently active test case, -1 means "none yet".
        self.current_index = -1
        # Collected responses for this run; written to JSON when finished.
        self.responses: list[dict] = []
        # Flag + index for the currently viewing item. While an item is in its
        # 3–5 second viewing phase, a timer will fire to hide the text. The
        # index is used so that stale timers never affect newer items.
        self._viewing_active = False
        self._viewing_index: Optional[int] = None

        # Timer used for the on-screen countdown while a word is visible.
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(100)  # update roughly every 0.1s
        self._countdown_timer.timeout.connect(self._on_countdown_tick)
        self._remaining_ms = 0

        self._build_ui()
        self._next_case()

    # --- Test case generation ---------------------------------------------------------
    def _build_test_cases(self) -> list[dict]:
        """Create a multi-level set of test items for the vision check.

        Each case describes the target text, its difficulty level, the
        font size, blur radius (for QGraphicsBlurEffect on the label), and
        a per-item display duration between 3 and 5 seconds. The items are
        generated up-front so they can be logged in detail.
        """

        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        words = [
            "CAT",
            "DOG",
            "TREE",
            "BOOK",
            "SCREEN",
            "HEALTH",
            "VISION",
            "LIGHT",
            "FOCUS",
            "BLINK",
        ]

        levels = [
            {
                "name": "Level 1 – Easy",
                "font_size": 40,
                "blur_radius": 0.0,
                "items": 4,
                "kind": "letter",
            },
            {
                "name": "Level 2 – Medium",
                "font_size": 28,
                "blur_radius": 1.5,
                "items": 4,
                "kind": "word",
            },
            {
                "name": "Level 3 – Hard",
                "font_size": 20,
                "blur_radius": 3.0,
                "items": 4,
                "kind": "word",
            },
        ]

        cases: list[dict] = []
        for level in levels:
            for _ in range(level["items"]):
                # Choose either a single letter or a short word
                if level["kind"] == "letter":
                    text = random.choice(letters)
                else:
                    text = random.choice(words)

                # Each item is visible for a random duration between 3–5 seconds
                duration_sec = random.uniform(3.0, 5.0)

                cases.append(
                    {
                        "text": text,
                        "level_name": level["name"],
                        "font_size": level["font_size"],
                        "blur_radius": level["blur_radius"],
                        "duration_sec": float(duration_sec),
                    }
                )
        return cases

    # --- UI ---------------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.resize(480, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        desc = QLabel(
            "Look at the letter or word below and type exactly what you see.\n"
            "Items will become smaller and/or blurrier as the test progresses.\n\n"
            "This is only a rough self-check and *not* a medical diagnosis."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        # Shows a short countdown while the item is still visible.
        self.timer_label = QLabel("")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timer_label)

        self.text_label = QLabel("")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Start with a reasonable default font; we will adjust size per case.
        self.text_label.setFont(QFont("Segoe UI", 32))
        layout.addWidget(self.text_label, stretch=1)

        self.input_edit = QLineEdit()
        # Keep the text box always enabled so it feels responsive. The user
        # can type once the text has appeared; the test will still hide the
        # text after 3–5 seconds to encourage recall from memory.
        self.input_edit.setPlaceholderText(
            "After the text disappears, type exactly what you saw here..."
        )
        # Limit length a bit so users don't accidentally paste very long text.
        self.input_edit.setMaxLength(16)
        layout.addWidget(self.input_edit)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        # Skip button for items the user cannot read clearly.
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._skip_current)
        button_row.addWidget(self.skip_btn)

        self.submit_btn = QPushButton("Submit answer")
        self.submit_btn.clicked.connect(self._record_response)
        button_row.addWidget(self.submit_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close)
        button_row.addWidget(close_btn)

        self.input_edit.returnPressed.connect(self._record_response)

    # --- Flow -------------------------------------------------------------------------
    def _next_case(self) -> None:
        """Advance to the next test case and start its viewing timer.

        For each case, the text is first shown for 3–5 seconds while the
        answer field is disabled. After the viewing period, the text is
        cleared and the user can type what they remember.
        """

        self.current_index += 1
        if self.current_index >= len(self.test_cases):
            self._finish_test()
            return

        case = self.test_cases[self.current_index]

        # Update label text and font size
        font = QFont("Segoe UI", case["font_size"])
        self.text_label.setFont(font)
        self.text_label.setText(case["text"])

        # Apply blur effect according to level difficulty
        blur_radius = float(case["blur_radius"])
        if blur_radius > 0:
            effect = QGraphicsBlurEffect(self)
            effect.setBlurRadius(blur_radius)
            self.text_label.setGraphicsEffect(effect)
        else:
            self.text_label.setGraphicsEffect(None)

        self.progress_label.setText(
            f"{case['level_name']} – item {self.current_index + 1} of {len(self.test_cases)}"
        )

        # Reset and enable the input for the new item.
        self.input_edit.clear()
        self.input_edit.setEnabled(True)
        self.submit_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)
        self.input_edit.setFocus()

        # Stop any previous countdown and start a fresh one for this case.
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        duration_sec = float(case.get("duration_sec", 3.0))
        duration_sec = max(3.0, min(5.0, duration_sec))
        self._remaining_ms = int(duration_sec * 1000)
        self._update_timer_label()

        # Start the timed viewing phase for this specific item.
        self._viewing_active = True
        self._viewing_index = self.current_index

        index_for_timer = self.current_index
        QTimer.singleShot(
            self._remaining_ms,
            lambda idx=index_for_timer: self._end_viewing_current_case(idx),
        )
        self._countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        """Update the on-screen countdown while a word is visible."""

        if not self._viewing_active or self._viewing_index is None:
            if self._countdown_timer.isActive():
                self._countdown_timer.stop()
            self.timer_label.setText("")
            return

        self._remaining_ms = max(0, self._remaining_ms - self._countdown_timer.interval())
        self._update_timer_label()

        if self._remaining_ms <= 0:
            if self._countdown_timer.isActive():
                self._countdown_timer.stop()

    def _update_timer_label(self) -> None:
        """Render the current remaining time into the timer label."""

        seconds = self._remaining_ms / 1000.0
        # Show with a single decimal place for a smooth countdown.
        if self._remaining_ms > 0:
            self.timer_label.setText(f"Time left: {seconds:.1f} s")
        else:
            self.timer_label.setText("")

    def _end_viewing_current_case(self, index: int) -> None:
        """End the viewing phase and hide the text for a specific item.

        The index argument ties this callback to the case that scheduled it,
        so that answers submitted early do not cause the *next* item to be
        cleared when the old timer fires.
        """

        # If the dialog is closing or this item is no longer active, do nothing.
        if not self._viewing_active:
            return
        if self._viewing_index is None or index != self._viewing_index:
            return
        if self.current_index != index:
            return

        self._viewing_active = False
        self._viewing_index = None

        # Stop the visible countdown and clear the timer label.
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self.timer_label.setText("")

        # Clear the text so the answer is based on recall, not direct copying.
        self.text_label.setText("")

    def _record_response(self) -> None:
        """Store the current answer and move to the next case."""

        if self.current_index < 0 or self.current_index >= len(self.test_cases):
            return

        case = self.test_cases[self.current_index]
        user_text = self.input_edit.text().strip().upper()
        if not user_text:
            # Do not advance on an empty answer; this keeps the test from
            # feeling "broken" if the user accidentally presses Enter.
            QMessageBox.warning(
                self,
                "Vision self-assessment",
                "Please type what you saw before submitting your answer.",
            )
            self.input_edit.setFocus()
            return
        target_text = str(case["text"]).strip().upper()
        is_correct = bool(user_text) and (user_text == target_text)

        self.responses.append(
            {
                "text": case["text"],
                "level_name": case["level_name"],
                "font_size": case["font_size"],
                "blur_radius": case["blur_radius"],
                "duration_sec": float(case.get("duration_sec", 3.0)),
                "user_input": user_text,
                "correct": bool(is_correct),
                "skipped": False,
            }
        )

        # Mark the viewing phase for this item as finished so that if the
        # display timer fires later, it will not affect subsequent items.
        self._viewing_active = False
        self._viewing_index = None

        self._next_case()

    def _skip_current(self) -> None:
        """Skip the current item when the user cannot read it clearly."""

        if self.current_index < 0 or self.current_index >= len(self.test_cases):
            return

        case = self.test_cases[self.current_index]

        # Record a skipped item as incorrect, with an empty user input and a
        # "skipped" flag so it can be analyzed separately if desired.
        self.responses.append(
            {
                "text": case["text"],
                "level_name": case["level_name"],
                "font_size": case["font_size"],
                "blur_radius": case["blur_radius"],
                "duration_sec": float(case.get("duration_sec", 3.0)),
                "user_input": "",
                "correct": False,
                "skipped": True,
            }
        )

        # Stop any viewing/countdown state for this item.
        self._viewing_active = False
        self._viewing_index = None
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self.timer_label.setText("")
        self.text_label.setText("")

        self._next_case()

    # --- Scoring & summary ------------------------------------------------------------
    def _finish_test(self) -> None:
        """Compute scores, log to JSON, and show a summary dialog."""

        if not self.responses:
            self.reject()
            return

        # Ensure no timers are still running once the test is complete.
        self._viewing_active = False
        self._viewing_index = None
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self.timer_label.setText("")

        total_items = len(self.responses)
        correct_items = sum(1 for r in self.responses if r["correct"])
        clarity_pct = round((correct_items / total_items) * 100.0, 1)

        # Per-level breakdown for a more informative result
        per_level: dict[str, dict[str, float]] = {}
        for r in self.responses:
            level = r["level_name"]
            info = per_level.setdefault(level, {"total": 0, "correct": 0})
            info["total"] += 1
            if r["correct"]:
                info["correct"] += 1

        for level, info in per_level.items():
            if info["total"] > 0:
                info["accuracy_pct"] = round(
                    (info["correct"] / info["total"]) * 100.0,
                    1,
                )
            else:
                info["accuracy_pct"] = 0.0

        if clarity_pct >= 80.0:
            classification = "Good"
            recommendation = (
                "Your ability to read these items was generally good. Keep "
                "following healthy screen habits like regular breaks and "
                "proper lighting."
            )
        elif clarity_pct >= 50.0:
            classification = "Average"
            recommendation = (
                "You had some difficulty reading smaller or blurrier items. "
                "Consider checking your viewing distance, screen brightness, "
                "and taking more frequent breaks."
            )
        else:
            classification = "Poor"
            recommendation = (
                "You found many items hard to read. This self-check cannot "
                "diagnose problems, but it may be a good idea to schedule a "
                "professional eye examination."
            )

        summary = {
            "total_items": total_items,
            "correct_items": correct_items,
            "clarity_percentage": clarity_pct,
            "classification": classification,
            "per_level": per_level,
        }

        entry = {
            "type": "vision_test",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "responses": self.responses,
            "summary": summary,
        }
        self.logger.append_entry(entry)

        # Build a human-readable per-level string
        level_lines = []
        for level, info in per_level.items():
            level_lines.append(
                f"{level}: {info['correct']} / {info['total']} correct "
                f"({info['accuracy_pct']}%)"
            )
        level_block = "\n".join(level_lines)

        message = (
            "Vision self-assessment results:\n\n"
            f"Correct items: {correct_items} / {total_items}\n"
            f"Overall clarity score: {clarity_pct}%\n"
            f"Classification: {classification}\n\n"
            "Per-level details:\n"
            f"{level_block}\n\n"
            f"Recommendation: {recommendation}\n\n"
            "Disclaimer: This is *not* a medical diagnosis. If you have any "
            "concerns about your vision, please consult a qualified "
            "eye-care professional."
        )

        QMessageBox.information(self, "Vision self-assessment", message)
        self.accept()

    def _on_close(self) -> None:
        """Close the dialog without saving if user cancels mid-test."""

        # Stop any in-progress viewing phase so timers do not try to
        # modify the UI after the dialog has been closed.
        self._viewing_active = False
        self._viewing_index = None
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        if hasattr(self, "timer_label"):
            self.timer_label.setText("")

        if self.responses:
            reply = QMessageBox.question(
                self,
                "Close test",
                "Close now? Your current test run will not be saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.reject()


class EmojiTrayWindow(QWidget):
    """Small always-on-top emoji window used for background mode."""

    def __init__(self, main_window: "VisionMateWindow") -> None:
        super().__init__(parent=main_window)
        self.main_window = main_window

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        width, height = 80, 80
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None else None
        if geo is not None:
            x = geo.right() - width - 10
            y = geo.top() + 10
        else:
            x, y = 50, 50
        self.setGeometry(x, y, width, height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.emoji_label = QLabel(self.main_window.current_emoji)
        self.emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Transparent background so only the emoji is visible
        self.emoji_label.setStyleSheet(
            "QLabel { font-size: 28px; background-color: transparent; }"
        )
        layout.addWidget(self.emoji_label)

        # Drag state
        self._dragging = False
        self._drag_offset = None

    def mousePressEvent(self, event) -> None:  # noqa: D401
        """Start drag or handle click for restoring the main window."""

        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._click_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        """Allow the emoji window to be dragged around the screen."""

        if self._dragging and self._drag_offset is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            self.move(new_pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        """If mouse was not moved significantly, treat as a click to restore."""

        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            release_pos = event.globalPosition().toPoint()

            if hasattr(self, "_click_pos"):
                distance = (release_pos - self._click_pos).manhattanLength()
                # Small movement = click -> restore main window
                if distance < 5:
                    self.main_window.restore_from_tray()

    def set_emoji(self, emoji: str) -> None:
        self.emoji_label.setText(emoji)

    def show_bubble(self, message: str, duration_ms: int = 5000) -> None:
        """Show a tooltip-style bubble with the break message near the emoji."""

        # Position the tooltip roughly at the emoji location
        global_pos = self.emoji_label.mapToGlobal(self.emoji_label.rect().center())
        QToolTip.showText(global_pos, message, self, self.rect(), duration_ms)


class BreakOverlayWindow(QWidget):
    """Fullscreen, always-on-top overlay used for forced eye breaks.

    It darkens the screen, shows a rest message and a countdown, and then
    closes automatically after a fixed duration. The user can press ESC twice
    quickly as an emergency shortcut to dismiss it early.
    """

    closed = pyqtSignal()

    def __init__(self, duration_seconds: int = 30, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)

        self._duration_seconds = max(1, int(duration_seconds))
        self._remaining_seconds = self._duration_seconds
        self._last_esc_time: Optional[float] = None

        # Fullscreen, borderless, always on top.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        # Block interaction with the rest of the application while visible.
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Dark, semi-transparent background so the screen looks "disabled".
        self.setStyleSheet("background-color: rgba(15, 23, 42, 220);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        layout.addStretch(1)

        self._title_label = QLabel("Time to rest your eyes ")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Segoe UI", 28)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        self._message_label = QLabel("Look away from the screen for 30 seconds")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg_font = QFont("Segoe UI", 18)
        self._message_label.setFont(msg_font)
        layout.addWidget(self._message_label)

        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        countdown_font = QFont("Segoe UI", 22)
        self._countdown_label.setFont(countdown_font)
        layout.addWidget(self._countdown_label)

        self._hint_label = QLabel("Press ESC twice quickly to dismiss if needed")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_font = QFont("Segoe UI", 12)
        self._hint_label.setFont(hint_font)
        layout.addWidget(self._hint_label)

        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

    def showEvent(self, event) -> None:  # noqa: D401
        """Start the countdown when the overlay becomes visible."""

        super().showEvent(event)
        self._remaining_seconds = self._duration_seconds
        self._update_countdown_label()
        if not self._timer.isActive():
            self._timer.start()
        # Optional: soft sound at break start.
        QApplication.beep()

    def closeEvent(self, event) -> None:  # noqa: D401
        """Ensure timer is stopped and notify listeners when closing."""

        if self._timer.isActive():
            self._timer.stop()
        self.closed.emit()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        """Handle ESC key as an emergency double-press exit."""

        if event.key() == Qt.Key.Key_Escape:
            now = time.time()
            if self._last_esc_time is not None and (now - self._last_esc_time) < 1.5:
                # Second ESC within 1.5 seconds -> dismiss overlay.
                self.close()
                return
            self._last_esc_time = now
        else:
            super().keyPressEvent(event)

    def _on_tick(self) -> None:
        self._remaining_seconds -= 1
        if self._remaining_seconds <= 0:
            self._update_countdown_label()
            # Optional: soft sound at break end.
            QApplication.beep()
            self.close()
        else:
            self._update_countdown_label()

    def _update_countdown_label(self) -> None:
        self._countdown_label.setText(f"Break ends in {self._remaining_seconds} s")


class VisionMateWindow(QWidget):
    """Main desktop application window and controller (PyQt)."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("VisionMate - Eye Health Monitor")
        self.resize(560, 320)

        self.event_queue: "queue.Queue[tuple]" = queue.Queue()
        self.notifier = NotificationManager()
        self.logger = UsageLogger()

        self.monitor_thread: Optional[MonitorThread] = None
        self.monitoring_active = False
        self.tray_window: Optional[EmojiTrayWindow] = None
        self.stats_page: Optional[StatsDashboardWindow] = None
        self.eye_test_page: Optional[EyeTestPage] = None
        self.break_overlay: Optional[BreakOverlayWindow] = None

        self.current_emoji = "😴"

        self.status_text = "Monitoring: OFF"
        self.distance_text = "Distance: No face detected"
        self.alert_text = "Last alert: None"
        self.blink_text = "Blink rate: --"
        self.posture_text = "Posture: Unknown"

        # Break reminder interval in seconds (default ~35 minutes)
        self.break_seconds: float = 2100.0

        self._build_ui()
        self._apply_styles()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_events)
        self.timer.start(500)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #020617;
                color: #e5e7eb;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QLabel#titleLabel {
                font-size: 18px;
                font-weight: 600;
                color: #f9fafb;
            }
            QLabel#statusLabel {
                font-size: 13px;
                font-weight: 600;
            }
            QFrame#card {
                background-color: #020617;
                border-radius: 16px;
                border: 1px solid #1f2937;
            }
            QPushButton {
                background-color: #111827;
                border-radius: 8px;
                padding: 6px 14px;
                color: #e5e7eb;
            }
            QPushButton:hover {
                background-color: #1f2937;
            }
            QPushButton:pressed {
                background-color: #030712;
            }
            QPushButton#exitButton {
                background-color: #b91c1c;
            }
            QPushButton#exitButton:hover {
                background-color: #dc2626;
            }
            QLabel#bubbleLabel {
                background-color: #111827;
                border-radius: 12px;
                border: 1px solid #4b5563;
                padding: 4px 8px;
            }
            """
        )

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.stack = QStackedWidget(self)
        root_layout.addWidget(self.stack)

        # --- Main monitoring dashboard page -----------------------------------------
        self.main_page = QWidget()
        main_layout = QVBoxLayout(self.main_page)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        title_label = QLabel("VisionMate – Eye Health Monitor")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(6)

        self.status_label = QLabel(self.status_text)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.status_label)

        self.emoji_label = QLabel(self.current_emoji)
        emoji_font = QFont("Segoe UI Emoji", 48)
        self.emoji_label.setFont(emoji_font)
        self.emoji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.emoji_label)

        self.bubble_label = QLabel("")
        self.bubble_label.setObjectName("bubbleLabel")
        self.bubble_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bubble_label.setWordWrap(True)
        self.bubble_label.hide()
        card_layout.addWidget(self.bubble_label)

        self.distance_label = QLabel(self.distance_text)
        self.distance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.distance_label)

        # Live blink-rate display (updated from background blink detector).
        self.blink_label = QLabel(self.blink_text)
        self.blink_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.blink_label)

        # Live posture status (green for good, red for poor posture).
        self.posture_label = QLabel(self.posture_text)
        self.posture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.posture_label)

        self.alert_label = QLabel(self.alert_text)
        self.alert_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.alert_label)

        main_layout.addWidget(card)

        settings_row = QHBoxLayout()
        main_layout.addLayout(settings_row)

        break_label = QLabel("Break reminder every:")
        settings_row.addWidget(break_label)

        self.break_hours_spin = QSpinBox()
        self.break_hours_spin.setRange(0, 23)
        self.break_hours_spin.setSuffix(" h")
        settings_row.addWidget(self.break_hours_spin)

        self.break_minutes_spin = QSpinBox()
        self.break_minutes_spin.setRange(0, 59)
        self.break_minutes_spin.setSuffix(" m")
        settings_row.addWidget(self.break_minutes_spin)

        self.break_seconds_spin = QSpinBox()
        self.break_seconds_spin.setRange(0, 59)
        self.break_seconds_spin.setSuffix(" s")
        settings_row.addWidget(self.break_seconds_spin)

        total = int(self.break_seconds)
        hours = total // 3600
        remainder = total % 3600
        minutes = remainder // 60
        seconds = remainder % 60
        self.break_hours_spin.setValue(hours)
        self.break_minutes_spin.setValue(minutes)
        self.break_seconds_spin.setValue(seconds)

        self.break_hours_spin.valueChanged.connect(self._on_break_time_changed)
        self.break_minutes_spin.valueChanged.connect(self._on_break_time_changed)
        self.break_seconds_spin.valueChanged.connect(self._on_break_time_changed)

        button_row = QHBoxLayout()
        main_layout.addLayout(button_row)

        self.start_btn = QPushButton("Start monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        button_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        button_row.addWidget(self.stop_btn)

        self.bg_btn = QPushButton("Run in background")
        self.bg_btn.clicked.connect(self.run_in_background_mode)
        button_row.addWidget(self.bg_btn)

        bottom_row = QHBoxLayout()
        main_layout.addLayout(bottom_row)

        eye_test_btn = QPushButton("Start Eye Test")
        eye_test_btn.clicked.connect(self.open_vision_test)
        bottom_row.addWidget(eye_test_btn)

        stats_btn = QPushButton("Statistics Dashboard")
        stats_btn.clicked.connect(self.open_stats_dashboard)
        bottom_row.addWidget(stats_btn)

        exit_btn = QPushButton("Exit")
        exit_btn.setObjectName("exitButton")
        exit_btn.clicked.connect(self.close)
        bottom_row.addWidget(exit_btn)

        info_label = QLabel(
            "VisionMate monitors your distance from the screen and reminds you to "
            "take breaks after long continuous viewing."
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        self.stack.addWidget(self.main_page)
        self.stack.setCurrentWidget(self.main_page)

    # --- Helper methods for UI state ---
    def _set_status(self, text: str) -> None:
        self.status_text = text
        self.status_label.setText(text)

    def _set_distance(self, text: str) -> None:
        self.distance_text = text
        self.distance_label.setText(text)

    def _set_alert(self, text: str) -> None:
        self.alert_text = text
        self.alert_label.setText(text)

    def _set_blink(self, text: str) -> None:
        self.blink_text = text
        self.blink_label.setText(text)

    def _set_posture(self, text: str, *, state: str | None = None) -> None:
        """Update posture text and colour based on state ('good'/'poor')."""

        self.posture_text = text
        self.posture_label.setText(text)

        # Colour feedback: green for good, red for poor, neutral otherwise.
        colour = None
        key = (state or "").lower()
        if key == "good":
            colour = "#10b981"  # emerald-500
        elif key == "poor":
            colour = "#ef4444"  # red-500

        if colour is not None:
            self.posture_label.setStyleSheet(f"color: {colour};")
        else:
            self.posture_label.setStyleSheet("")

    def _set_emoji(self, emoji: str) -> None:
        self.current_emoji = emoji
        self.emoji_label.setText(emoji)
        if self.tray_window is not None:
            self.tray_window.set_emoji(emoji)

    def _show_bubble(self, message: str, duration_ms: int = 5000) -> None:
        if hasattr(self, "bubble_label") and self.bubble_label is not None:
            self.bubble_label.setText(message)
            self.bubble_label.show()
            QTimer.singleShot(duration_ms, self._hide_bubble)

        # Also show a bubble near the tray emoji when in background mode
        if self.tray_window is not None:
            try:
                self.tray_window.show_bubble(message, duration_ms=duration_ms)
            except Exception:  # noqa: BLE001
                pass

    def _hide_bubble(self) -> None:
        if hasattr(self, "bubble_label") and self.bubble_label is not None:
            self.bubble_label.hide()

    def show_break_overlay(self, duration_seconds: int = 30) -> None:
        """Show the fullscreen forced-break overlay if not already visible."""

        if self.break_overlay is not None and self.break_overlay.isVisible():
            return

        # Create as a top-level window (no parent) so showFullScreen() covers
        # the entire screen and is not clipped to the main window.
        self.break_overlay = BreakOverlayWindow(duration_seconds=duration_seconds)
        self.break_overlay.closed.connect(self._on_break_overlay_closed)
        self.break_overlay.showFullScreen()
        self.break_overlay.raise_()
        self.break_overlay.activateWindow()

    def _on_break_overlay_closed(self) -> None:
        """Clear reference when the forced-break overlay is closed."""

        self.break_overlay = None

    def _on_break_time_changed(self, _value: int) -> None:
        """Update the break reminder interval from hours/minutes/seconds."""

        hours = self.break_hours_spin.value()
        minutes = self.break_minutes_spin.value()
        seconds = self.break_seconds_spin.value()
        total = hours * 3600 + minutes * 60 + seconds

        # Ensure we don't end up with zero; fall back to 60 seconds
        if total <= 0:
            total = 60
            self.break_hours_spin.blockSignals(True)
            self.break_minutes_spin.blockSignals(True)
            self.break_seconds_spin.blockSignals(True)
            self.break_hours_spin.setValue(0)
            self.break_minutes_spin.setValue(1)
            self.break_seconds_spin.setValue(0)
            self.break_hours_spin.blockSignals(False)
            self.break_minutes_spin.blockSignals(False)
            self.break_seconds_spin.blockSignals(False)

        self.break_seconds = float(total)
        if self.monitor_thread is not None:
            self.monitor_thread.break_interval_seconds = float(total)

    # --- Monitoring controls ---
    def start_monitoring(self) -> None:
        if self.monitoring_active and self.monitor_thread is not None:
            return

        try:
            self.monitor_thread = MonitorThread(
                self.event_queue,
                break_interval_seconds=self.break_seconds,
            )
            self.monitor_thread.start()
            self.monitoring_active = True
            self._set_status("Monitoring: ON")
            self._set_emoji("😊")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "VisionMate",
                f"Could not start monitoring:\n{exc}",
            )

    def stop_monitoring(self) -> None:
        if not self.monitoring_active or self.monitor_thread is None:
            return

        thread = self.monitor_thread
        self.monitoring_active = False
        self.monitor_thread = None

        try:
            thread.stop()
            thread.join(timeout=3.0)
        except Exception:  # noqa: BLE001
            pass

        try:
            stats = thread.get_session_stats()
            self.logger.append_entry(stats)
        except Exception:  # noqa: BLE001
            pass

        self._set_status("Monitoring: OFF")
        self._set_distance("Distance: No face detected")
        self._set_emoji("😴")

    def open_vision_test(self) -> None:
        """Switch to the full-page EyeTestPage and start a new test run."""

        if self.eye_test_page is None:
            self.eye_test_page = EyeTestPage(self.logger, parent=self)
            if hasattr(self, "stack"):
                # Allow the eye test page to request going back to the main dashboard.
                try:
                    self.eye_test_page.back_requested.connect(self.show_main_dashboard)
                except Exception:
                    # Older versions of EyeTestPage may not define back_requested;
                    # in that case, navigation will still work via direct calls.
                    pass
                self.stack.addWidget(self.eye_test_page)

        if hasattr(self, "stack") and self.eye_test_page is not None:
            self.stack.setCurrentWidget(self.eye_test_page)
            # Start a fresh test session every time the page is opened
            self.eye_test_page.start_test()

    def show_main_dashboard(self) -> None:
        if hasattr(self, "stack") and hasattr(self, "main_page"):
            self.stack.setCurrentWidget(self.main_page)

    def open_stats_dashboard(self) -> None:
        """Open the Statistics Dashboard window, creating it on first use.

        The dashboard reads data from usage_log.json and shows charts for
        screen distance, screen time, and vision test progress. If the
        window already exists, it is simply raised and activated.
        """

        if self.stats_page is None:
            self.stats_page = StatsDashboardWindow(parent=self)
            if hasattr(self, "stack"):
                self.stats_page.back_requested.connect(self.show_main_dashboard)
                self.stack.addWidget(self.stats_page)

        if hasattr(self, "stack") and self.stats_page is not None:
            self.stack.setCurrentWidget(self.stats_page)

    # --- Background (tray) mode ---
    def run_in_background_mode(self) -> None:
        """Hide main window and show a tiny always-on-top emoji window.

        Monitoring continues (and is auto-started if it was not already running).
        Clicking the emoji brings the full GUI back.
        """

        if not self.monitoring_active or self.monitor_thread is None:
            self.start_monitoring()

        if self.tray_window is not None:
            return

        self.hide()
        self.tray_window = EmojiTrayWindow(self)
        self.tray_window.show()

    def restore_from_tray(self) -> None:
        if self.tray_window is not None:
            self.tray_window.close()
            self.tray_window = None

        self.showNormal()
        self.raise_()
        self.activateWindow()

    # --- Event processing ---
    def _process_events(self) -> None:
        try:
            while True:
                event_name, payload = self.event_queue.get_nowait()

                if event_name == "distance_update":
                    distance = payload.get("distance_cm")
                    if distance is None:
                        self._set_distance("Distance: No face detected")
                        self._set_emoji("😐")
                    else:
                        self._set_distance(f"Distance: {distance:.1f} cm")

                        threshold = (
                            self.monitor_thread.min_distance_cm
                            if self.monitor_thread is not None
                            else 40.0
                        )
                        if distance < threshold:
                            self._set_emoji("😡")
                        else:
                            self._set_emoji("😊")

                elif event_name == "distance_warning":
                    distance = payload.get("distance_cm")
                    msg = (
                        "You are too close to the screen. "
                        f"Estimated distance: ≈ {distance} cm. Please sit back a bit."
                    )
                    self._set_alert("Last alert: Too close to screen")
                    self.notifier.notify(
                        "VisionMate – Distance warning",
                        msg,
                    )

                elif event_name == "break_reminder":
                    self._set_alert("Last alert: Break reminder")

                    # Show a cloud/thought-bubble style reminder near the emoji
                    total = int(self.break_seconds)
                    hours = total // 3600
                    remainder = total % 3600
                    minutes = remainder // 60
                    seconds = remainder % 60

                    time_parts = []
                    if hours:
                        time_parts.append(f"{hours}h")
                    if minutes:
                        time_parts.append(f"{minutes}m")
                    if seconds or not time_parts:
                        time_parts.append(f"{seconds}s")

                    human_time = " ".join(time_parts)

                    bubble_msg = (
                        f"{self.current_emoji} 💭\n"
                        f"You've been looking at the screen for {human_time}.\n"
                        "Time to take a short eye break!"
                    )
                    self._show_bubble(bubble_msg, duration_ms=7000)

                    self.notifier.notify(
                        "VisionMate – Time for a break",
                        (
                            "You have been looking at the screen continuously for a long "
                            "time. Please take a short break to rest your eyes."
                        ),
                        timeout=10,
                    )

                    # Also show the fullscreen forced-break overlay for 30 seconds.
                    self.show_break_overlay(duration_seconds=30)

                elif event_name == "blink_rate_update":
                    bpm = payload.get("blink_rate_bpm")
                    if bpm is None:
                        self._set_blink("Blink rate: --")
                    else:
                        self._set_blink(f"Blink rate: {float(bpm):.1f} blinks/min")

                elif event_name == "blink_low":
                    bpm = payload.get("blink_rate_bpm")
                    approx = f"≈ {float(bpm):.1f}" if bpm is not None else "low"
                    self._set_alert("Last alert: Low blink rate")
                    # Hint to the user that their eyes may be getting dry.
                    self.notifier.notify(
                        "VisionMate – Blink reminder",
                        (
                            f"Your blink rate seems low ({approx} blinks/min). "
                            "Try to blink more often or take a short eye break."
                        ),
                    )
                    # Optional visual cue: sleepy emoji.
                    self._set_emoji("😴")

                elif event_name == "posture_update":
                    state = (payload.get("state") or "").lower()
                    if state == "good":
                        self._set_posture("Posture: Good", state="good")
                    elif state == "poor":
                        self._set_posture("Posture: Poor", state="poor")
                    else:
                        self._set_posture("Posture: Unknown", state=None)

                elif event_name == "posture_warning":
                    state = (payload.get("state") or "").lower()
                    self._set_alert("Last alert: Posture warning")
                    msg = "Try sitting upright to reduce strain on your neck and back."
                    self.notifier.notify("VisionMate – Posture reminder", msg)
                    # Emphasise poor posture in the label colour.
                    if state == "poor":
                        self._set_posture("Posture: Poor", state="poor")

                elif event_name == "error":
                    self._set_alert("Last alert: Error")
                    message = payload.get("message", "Unknown error")
                    QMessageBox.critical(self, "VisionMate", message)
                    if self.monitoring_active:
                        self.stop_monitoring()
        except queue.Empty:
            pass

    def closeEvent(self, _event) -> None:  # noqa: D401
        """Ensure monitoring and tray window are cleaned up on close."""

        if self.monitoring_active:
            self.stop_monitoring()

        if self.tray_window is not None:
            self.tray_window.close()
            self.tray_window = None


if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    window = VisionMateWindow()
    window.show()
    sys.exit(qt_app.exec())
