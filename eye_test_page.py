from __future__ import annotations

import random
from datetime import datetime
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGraphicsBlurEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# Fixed viewing duration (in seconds) for each question.
QUESTION_DURATION_SEC: float = 5.0

# Simple string constants for the test state machine.
STATE_IDLE = "IDLE"
STATE_SHOWING_QUESTION = "SHOWING_QUESTION"
STATE_WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
STATE_RECORDING_RESULT = "RECORDING_RESULT"
STATE_FINISHED = "FINISHED"


class EyeTestPage(QWidget):
    """Full-page vision self-test screen for VisionMate.

    This widget is designed to be used as a page inside a QStackedWidget.
    It runs a mixed text + image based vision test, with timed questions,
    automatic progression, scoring, classification, and JSON logging via
    a UsageLogger instance supplied by the host application.
    """

    # Signal emitted when the page requests to return to the main dashboard.
    back_requested = pyqtSignal()

    def __init__(self, logger, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.logger = logger

        # Question set and results for the current run
        self._questions: List[dict] = []
        self._results: List[dict] = []
        self._current_index: int = -1
        self._question_active: bool = False
        self._remaining_ms: int = 0
        # Has the countdown for the current question reached zero?
        self._timeout_reached: bool = False
        # Explicit state for the test controller.
        self._state: str = STATE_IDLE
        # True when we are showing the end-of-test result screen instead of questions.
        self._in_result_screen: bool = False

        # Timer that drives the visible countdown and handles timeouts.
        # Use 1-second ticks for a clear, simple countdown.
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()

    # --- Public API -----------------------------------------------------------------
    def start_test(self) -> None:
        """Start a new test session with a freshly shuffled question set."""

        self._questions = self._build_question_set()
        self._results = []
        self._current_index = -1
        self._question_active = False
        self._remaining_ms = 0
        self._timeout_reached = False
        self._state = STATE_IDLE
        self._in_result_screen = False
        self._countdown_timer.stop()

        # Restore UI to question-answer mode
        self._progress_label.setText("")
        self._timer_label.setText("")
        self._content_label.clear()
        self._input_edit.clear()
        self._input_edit.setEnabled(True)
        self._input_edit.setVisible(True)
        self._skip_btn.setEnabled(True)
        self._skip_btn.setVisible(True)
        self._submit_btn.setEnabled(True)
        self._submit_btn.setVisible(True)
        self._exit_btn.setText("Exit test")
        print("[EyeTest] Starting new test session with", len(self._questions), "questions")

        self._next_question()

    # --- UI construction ------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("VisionMate – Eye Test")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        desc = QLabel(
            "You will see a series of short letters, words, or simple shapes.\n"
            "Each item is shown only briefly. Type what you saw, or skip if you\n"
            "could not read it clearly. This is a self-check only, not a medical test."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress_label)

        self._timer_label = QLabel("")
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._timer_label)

        # Central display area for text or images
        self._content_label = QLabel("")
        self._content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._content_label.setMinimumHeight(180)
        # Ensure very high contrast and large, readable text regardless of theme.
        base_font = QFont("Segoe UI", 40)
        self._content_label.setFont(base_font)
        self._content_label.setStyleSheet(
            "color: #f9fafb; background-color: #111827; border-radius: 8px;"
        )
        layout.addWidget(self._content_label, stretch=1)

        # Answer input
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Type what you saw here…")
        self._input_edit.setMaxLength(32)
        layout.addWidget(self._input_edit)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self._on_skip_clicked)
        button_row.addWidget(self._skip_btn)

        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit_clicked)
        button_row.addWidget(self._submit_btn)

        self._exit_btn = QPushButton("Exit test")
        self._exit_btn.clicked.connect(self._on_exit_clicked)
        button_row.addWidget(self._exit_btn)

        self._input_edit.returnPressed.connect(self._on_submit_clicked)

    # --- Question generation --------------------------------------------------------
    def _build_question_set(self) -> List[dict]:
        """Create an ordered set of **text-only** questions with difficulty.

        The test progresses from easy to medium to hard questions so that
        difficulty clearly increases as you move forward.
        """

        # Easy level: large letters and very short, clear words.
        easy_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        easy_words = ["CAT", "DOG", "EYE"]

        # Medium level: common short to medium words.
        medium_words = [
            "TREE",
            "BOOK",
            "REST",
            "BREAK",
            "LIGHT",
            "FOCUS",
        ]

        # Hard level: longer words that are more demanding to read when small/blurred.
        hard_words = [
            "SCREEN",
            "HEALTH",
            "VISION",
            "MOUSE",
            "LAPTOP",
        ]

        questions: List[dict] = []

        # Configure how many questions per level (total 10).
        num_easy = 4
        num_medium = 3
        num_hard = 3

        for _ in range(num_easy):
            # Mix of easy letters and very short words.
            if random.random() < 0.5 and easy_letters:
                target = random.choice(easy_letters)
            else:
                target = random.choice(easy_words)
            # Very large font, no blur: should be readable for most people.
            font_size = 54 if len(target) <= 2 else 46
            questions.append(
                {
                    "type": "text",
                    "difficulty": "Easy",
                    "target": target,
                    "font_size": font_size,
                    "blur_radius": 0.0,
                    "duration_sec": float(QUESTION_DURATION_SEC),
                }
            )

        for _ in range(num_medium):
            # Moderate font size, with a small blur to make the task tougher.
            target = random.choice(medium_words)
            font_size = 40
            blur_radius = 1.4
            questions.append(
                {
                    "type": "text",
                    "difficulty": "Medium",
                    "target": target,
                    "font_size": font_size,
                    "blur_radius": blur_radius,
                    "duration_sec": float(QUESTION_DURATION_SEC),
                }
            )

        for _ in range(num_hard):
            # Smaller font with stronger blur – these are intentionally challenging.
            target = random.choice(hard_words)
            font_size = 32
            blur_radius = 2.8
            questions.append(
                {
                    "type": "text",
                    "difficulty": "Hard",
                    "target": target,
                    "font_size": font_size,
                    "blur_radius": blur_radius,
                    "duration_sec": float(QUESTION_DURATION_SEC),
                }
            )

        # Do not shuffle here so that difficulty increases throughout the test.
        return questions

    # --- Per-question flow ----------------------------------------------------------
    def _next_question(self) -> None:
        """Advance to the next question or finish the test if done."""

        # Between questions, ensure timer is stopped and we are not waiting.
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self._question_active = False
        self._state = STATE_IDLE
        self._timer_label.setText("")
        self._timeout_reached = False

        self._current_index += 1
        if self._current_index >= len(self._questions):
            print("[EyeTest] All questions completed; entering FINISHED state")
            self._finish_test()
            return

        q = self._questions[self._current_index]
        total = len(self._questions)
        print(
            f"[EyeTest] Showing question {self._current_index + 1} of {total}: "
            f"{q.get('target')}"
        )

        # Update progress label
        self._progress_label.setText(
            f"Question {self._current_index + 1} of {len(self._questions)}"
        )

        # Reset UI state for the new question.
        self._input_edit.clear()
        self._input_edit.setEnabled(True)
        self._input_edit.setVisible(True)
        self._submit_btn.setEnabled(True)
        self._submit_btn.setVisible(True)
        self._skip_btn.setEnabled(True)
        self._skip_btn.setVisible(True)
        self._input_edit.setFocus()

        # Render question content (text only) while in SHOWING_QUESTION state.
        self._state = STATE_SHOWING_QUESTION
        self._show_question_content(q)

        # Start countdown for this question and then move to WAITING_FOR_INPUT.
        duration_sec = float(q.get("duration_sec", QUESTION_DURATION_SEC))
        self._remaining_ms = int(max(1.0, duration_sec) * 1000)
        self._question_active = True
        self._state = STATE_WAITING_FOR_INPUT
        self._update_timer_label()
        self._countdown_timer.start()

    def _show_question_content(self, q: dict) -> None:
        """Render the current question's text in the central label."""

        # Clear any previous blur effect.
        self._content_label.setGraphicsEffect(None)

        text = str(q.get("target", "")).strip().upper()
        if not text:
            text = "?"
        font_size = int(q.get("font_size", 40))
        font = QFont("Segoe UI", font_size)
        self._content_label.setFont(font)
        self._content_label.setText(text)

        blur_radius = float(q.get("blur_radius", 0.0))
        if blur_radius > 0:
            effect = QGraphicsBlurEffect(self)
            effect.setBlurRadius(blur_radius)
            self._content_label.setGraphicsEffect(effect)

    # --- Timers and answers ---------------------------------------------------------
    def _on_countdown_tick(self) -> None:
        """Handle the per-question countdown and mark timeout without advancing.

        A question still ends only when the user clicks Submit or Skip. When the
        timer reaches zero, we just remember that a timeout occurred so the
        eventual result can be tagged as from_timeout.
        """

        if self._state != STATE_WAITING_FOR_INPUT:
            if self._countdown_timer.isActive():
                self._countdown_timer.stop()
            return

        self._remaining_ms = max(
            0, self._remaining_ms - self._countdown_timer.interval()
        )
        self._update_timer_label()

        if self._remaining_ms <= 0:
            print(
                f"[EyeTest] Timer reached zero for question "
                f"{self._current_index + 1}"
            )
            self._countdown_timer.stop()
            self._remaining_ms = 0
            self._timeout_reached = True
            self._update_timer_label()

    def _update_timer_label(self) -> None:
        seconds = max(0, int(round(self._remaining_ms / 1000.0)))
        if self._remaining_ms > 0:
            self._timer_label.setText(f"Time left: {seconds} s")
        else:
            self._timer_label.setText("Time left: 0 s")

    def _on_submit_clicked(self) -> None:
        if self._state != STATE_WAITING_FOR_INPUT:
            return
        print(
            f"[EyeTest] Submit pressed at question {self._current_index + 1}"
        )
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        # If the timer had already reached zero, preserve that information.
        self._finalise_current_answer(timed_out=self._timeout_reached, manual=True)

    def _on_skip_clicked(self) -> None:
        if self._state != STATE_WAITING_FOR_INPUT:
            return
        print(
            f"[EyeTest] Skip pressed at question {self._current_index + 1}"
        )
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        # Manual skip: record as skipped, preserving whether a timeout occurred.
        self._record_result(skipped=True, from_timeout=self._timeout_reached)
        self._advance_after_answer()

    def _on_exit_clicked(self) -> None:
        print(f"[EyeTest] Exit pressed, state={self._state}")
        # If we are on the final result screen, always return immediately with
        # no confirmation – the test is already complete.
        if self._state == STATE_FINISHED or self._in_result_screen:
            if self._countdown_timer.isActive():
                self._countdown_timer.stop()
            self._question_active = False
            self._remaining_ms = 0
            self._timer_label.setText("")
            self._content_label.clear()
            self._progress_label.setText("")
            self._input_edit.clear()
            self._state = STATE_IDLE
            self._in_result_screen = False
            # Ask the host window to return to the dashboard.
            self.back_requested.emit()
            return

        if self._question_active or self._results:
            reply = QMessageBox.question(
                self,
                "Exit eye test",
                "Exit the test now? Current progress will not be saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Reset state and notify host window to switch back
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self._question_active = False
        self._remaining_ms = 0
        self._timer_label.setText("")
        self._content_label.clear()
        self._questions = []
        self._results = []
        self._current_index = -1
        self._state = STATE_IDLE
        self._in_result_screen = False
        # Ask the host window to return to the dashboard.
        self.back_requested.emit()

    def _finalise_current_answer(self, timed_out: bool, manual: bool = False) -> None:
        if not self._question_active:
            return

        text = self._input_edit.text().strip()
        if manual and not text:
            # On manual submit, require some input; suggest skip instead.
            QMessageBox.warning(
                self,
                "Eye test",
                "Please type what you saw, or press Skip if you could not read it.",
            )
            return

        if not text:
            # Programmatic calls (if any) with no text treat as skipped.
            self._record_result(skipped=True, from_timeout=timed_out)
        else:
            self._record_result(skipped=False, from_timeout=timed_out, user_text=text)

        self._advance_after_answer()

    def _record_result(
        self,
        *,
        skipped: bool,
        from_timeout: bool,
        user_text: Optional[str] = None,
    ) -> None:
        if self._current_index < 0 or self._current_index >= len(self._questions):
            return

        q = self._questions[self._current_index]
        target = str(q.get("target", "")).strip().upper()
        user_val = (user_text or "").strip().upper()

        is_correct = (not skipped) and bool(user_val) and (user_val == target.upper())

        result = {
            "index": self._current_index + 1,
            "type": q.get("type"),
            "difficulty": q.get("difficulty"),
            "target": target,
            "user_input": user_val,
            "duration_sec": float(q.get("duration_sec", 3.0)),
            "correct": bool(is_correct),
            "skipped": bool(skipped),
            "from_timeout": bool(from_timeout),
            "answer_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._results.append(result)
        print(
            "[EyeTest] Recorded result:",
            {
                "index": result["index"],
                "target": result["target"],
                "user_input": result["user_input"],
                "skipped": result["skipped"],
                "from_timeout": result["from_timeout"],
                "correct": result["correct"],
            },
        )

    def _advance_after_answer(self) -> None:
        self._question_active = False
        self._state = STATE_IDLE
        self._timeout_reached = False
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self._timer_label.setText("")
        # Keep the current question visible until the next one is shown.
        self._next_question()

    # --- Scoring and logging --------------------------------------------------------
    def _finish_test(self) -> None:
        # Ensure no timers are left running once the test is complete.
        self._question_active = False
        self._in_result_screen = True
        self._state = STATE_FINISHED
        if self._countdown_timer.isActive():
            self._countdown_timer.stop()
        self._timer_label.setText("")
        print("[EyeTest] Test finished; computing results")

        # Disable and hide input and the Submit/Skip buttons while showing results.
        self._input_edit.clear()
        self._input_edit.setEnabled(False)
        self._input_edit.setVisible(False)
        self._skip_btn.setEnabled(False)
        self._skip_btn.setVisible(False)
        self._submit_btn.setEnabled(False)
        self._submit_btn.setVisible(False)
        self._exit_btn.setText("Return to dashboard")

        # Results should always be crisp and easy to read, regardless of how
        # blurry the last question was.
        self._content_label.setGraphicsEffect(None)

        if not self._results:
            # Nothing answered; treat as cancelled, but still show a short
            # message in-page so the user does not see a blank screen.
            self._content_label.setText(
                "No answers were recorded for this test run.\n\n"
                "Press 'Return to dashboard' to go back to the main screen."
            )
            return

        total = len(self._results)
        correct = sum(1 for r in self._results if r["correct"])
        clarity_pct = round((correct / total) * 100.0, 1)

        if clarity_pct >= 80.0:
            classification = "Good"
            recommendation = (
                "Your ability to recognise these items was generally good. "
                "Keep following healthy screen habits like regular breaks and "
                "proper lighting."
            )
        elif clarity_pct >= 50.0:
            classification = "Average"
            recommendation = (
                "You had some difficulty with smaller or blurrier items. "
                "Consider checking your viewing distance, screen brightness, "
                "and taking more frequent breaks."
            )
        else:
            classification = "Poor"
            recommendation = (
                "Many items were hard to recognise. This self-check cannot "
                "diagnose problems, but it may be a good idea to schedule a "
                "professional eye examination."
            )

        # Aggregate results per difficulty label
        per_level: dict[str, dict[str, float]] = {}
        for r in self._results:
            level = str(r.get("difficulty") or "Unknown")
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

        summary = {
            "total_items": total,
            "correct_items": correct,
            "clarity_percentage": clarity_pct,
            "classification": classification,
            "per_level": per_level,
        }

        entry = {
            "type": "vision_test",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "responses": self._results,
            "summary": summary,
        }

        try:
            self.logger.append_entry(entry)
        except Exception:
            # Logging failures should not block showing the result
            pass

        # Human-readable message for the user
        level_lines = []
        for level, info in per_level.items():
            level_lines.append(
                f"{level}: {info['correct']} / {info['total']} correct "
                f"({info['accuracy_pct']}%)"
            )
        level_block = "\n".join(level_lines)

        message = (
            "Vision self-assessment results:\n\n"
            f"Correct items: {correct} / {total}\n"
            f"Overall clarity score: {clarity_pct}%\n"
            f"Classification: {classification}\n\n"
            "Per-level details:\n"
            f"{level_block}\n\n"
            f"Recommendation: {recommendation}\n\n"
            "Disclaimer: This is *not* a medical diagnosis. If you have any "
            "concerns about your vision, please consult a qualified "
            "eye-care professional."
        )

        # Show results directly on the Eye Test page instead of a popup,
        # so the user always sees a clear end-of-test screen.
        # Use a slightly smaller font for the summary block.
        summary_font = QFont("Segoe UI", 16)
        self._content_label.setFont(summary_font)
        self._content_label.setText(message)

        # The user can now press "Return to dashboard" (Exit button) to go back
        # to the main dashboard. We do not auto-switch pages here to avoid
        # briefly flashing a blank screen.
