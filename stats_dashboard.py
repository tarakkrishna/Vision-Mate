"""Statistics dashboard UI for VisionMate.

This module provides a PyQt-based window that visualises data from the
local usage_log.json file using matplotlib charts. It shows:

- Average screen distance (synthetic score) per day for the last 7 days
- Total screen time per day for the last 7 days
- Vision test clarity percentages over time

The heavy lifting of data loading and aggregation is delegated to
stats_data.py so that the UI module focuses on layout and chart
rendering only.
"""

from __future__ import annotations

import csv
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from stats_data import (
    DistancePoint,
    ScreenTimePoint,
    VisionTestPoint,
    compute_last_7_days_distance,
    compute_last_7_days_screen_time,
    compute_vision_test_progress,
    has_enough_data_for_dashboard,
)


class StatsDashboardWindow(QWidget):
    """A window that displays VisionMate statistics using matplotlib charts."""

    back_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VisionMate – Statistics Dashboard")
        # Slightly larger window so charts have space but still fit on
        # typical laptop screens.
        self.resize(860, 620)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        layout.addLayout(header_row)

        back_btn = QPushButton("\u2190 Back")
        back_btn.clicked.connect(self.back_requested.emit)
        header_row.addWidget(back_btn)

        header_row.addStretch(1)

        title = QLabel("Statistics Dashboard")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        header_row.addWidget(title)
        header_row.addStretch(1)

        if not has_enough_data_for_dashboard():
            msg = QLabel(
                "Not enough data yet. Use VisionMate for a few days to see insights."
            )
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
            return

        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)

        distance_tab = QWidget()
        distance_layout = QVBoxLayout(distance_tab)
        distance_layout.setContentsMargins(8, 8, 8, 8)
        distance_layout.setSpacing(10)
        self._populate_distance_tab(distance_layout)
        tabs.addTab(distance_tab, "Screen Distance")

        time_tab = QWidget()
        time_layout = QVBoxLayout(time_tab)
        time_layout.setContentsMargins(8, 8, 8, 8)
        time_layout.setSpacing(10)
        self._populate_time_tab(time_layout)
        tabs.addTab(time_tab, "Screen Time")

        vision_tab = QWidget()
        vision_layout = QVBoxLayout(vision_tab)
        vision_layout.setContentsMargins(8, 8, 8, 8)
        vision_layout.setSpacing(10)
        self._populate_vision_tab(vision_layout)
        tabs.addTab(vision_tab, "Vision Tests")

        # Row with an action button to export stats to a CSV file.
        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        button_row.addStretch(1)
        export_btn = QPushButton("Download stats5")
        export_btn.clicked.connect(self._export_stats)
        button_row.addWidget(export_btn)

    # --- Individual chart sections ----------------------------------------------------
    def _populate_distance_tab(self, layout: QVBoxLayout) -> None:
        points = compute_last_7_days_distance()
        if not points:
            msg = QLabel(
                "Not enough monitoring data to show screen distance trends yet."
            )
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
            return

        dates = [p.date for p in points]
        values = [p.average_distance_cm for p in points]

        # Use a card-style frame so the chart matches the main app theme.
        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(6)

        fig = Figure(figsize=(5, 3), constrained_layout=True)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ax = fig.add_subplot(111)

        ax.plot(dates, values, marker="o", color="#60a5fa")
        ax.set_title("Average screen distance (last 7 days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Distance score (higher is better)")
        ax.grid(True, alpha=0.2)
        fig.autofmt_xdate(rotation=30)

        card_layout.addWidget(canvas)
        layout.addWidget(card, stretch=1)

        info = QLabel(
            "This chart is an approximate score based on how often you sit too "
            "close to the screen. Fewer 'too close' alerts means a higher "
            "distance score."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

    def _populate_time_tab(self, layout: QVBoxLayout) -> None:
        points = compute_last_7_days_screen_time()
        if not points:
            msg = QLabel(
                "Not enough monitoring data to show screen time trends yet."
            )
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
            return

        dates = [p.date for p in points]
        values_hours = [p.total_seconds / 3600.0 for p in points]

        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(6)

        fig = Figure(figsize=(5, 3), constrained_layout=True)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ax = fig.add_subplot(111)

        ax.bar(dates, values_hours, color="#34d399")
        ax.set_title("Total screen time per day (last 7 days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hours of monitoring")
        ax.grid(axis="y", alpha=0.2)
        fig.autofmt_xdate(rotation=30)

        card_layout.addWidget(canvas)
        layout.addWidget(card, stretch=1)

        info = QLabel(
            "This chart shows how long VisionMate has been monitoring your "
            "screen use each day. Higher bars mean more continuous screen time."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

    def _populate_vision_tab(self, layout: QVBoxLayout) -> None:
        points = compute_vision_test_progress()
        if not points:
            msg = QLabel(
                "No vision test data yet. Run a few vision self-tests to see progress."
            )
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
            return

        dates = [p.timestamp for p in points]
        values = [p.clarity_percentage for p in points]

        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(6)

        fig = Figure(figsize=(5, 3), constrained_layout=True)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ax = fig.add_subplot(111)

        ax.plot(dates, values, marker="o", color="#fbbf24")
        ax.set_title("Vision test clarity over time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Clarity percentage (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)
        fig.autofmt_xdate(rotation=30)

        card_layout.addWidget(canvas)
        layout.addWidget(card, stretch=1)

        info = QLabel(
            "Each point represents a vision self-test. Higher percentages "
            "mean you correctly read more items at different difficulty levels."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

    # --- Export functionality ---------------------------------------------------------
    def _export_stats(self) -> None:
        """Export the current statistics to a CSV file chosen by the user.

        The CSV contains three logical sections:

        - screen_distance: one row per day with a synthetic distance score
        - screen_time: one row per day with total monitored hours
        - vision_test: one row per vision test with clarity percentage
        """

        # Ask the user where to save the file.
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save statistics",
            "visionmate_stats.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        # Recompute stats so the export matches what the charts show.
        distance_points = compute_last_7_days_distance()
        time_points = compute_last_7_days_screen_time()
        vision_points = compute_vision_test_progress()

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "section",
                        "timestamp_or_date",
                        "metric",
                        "value",
                    ]
                )

                for p in distance_points:
                    writer.writerow(
                        [
                            "screen_distance",
                            p.date.date().isoformat(),
                            "distance_score",
                            f"{p.average_distance_cm:.2f}",
                        ]
                    )

                for p in time_points:
                    writer.writerow(
                        [
                            "screen_time",
                            p.date.date().isoformat(),
                            "hours_monitored",
                            f"{p.total_seconds / 3600.0:.2f}",
                        ]
                    )

                for p in vision_points:
                    writer.writerow(
                        [
                            "vision_test",
                            p.timestamp.isoformat(timespec="seconds"),
                            "clarity_percentage",
                            f"{p.clarity_percentage:.1f}",
                        ]
                    )
        except OSError as exc:  # pragma: no cover - UI feedback only
            QMessageBox.critical(
                self,
                "Save statistics",
                f"Could not save statistics to file:\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Save statistics",
            "Statistics have been exported successfully.",
        )


# Manual test hook ---------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - for manual testing only
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = StatsDashboardWindow()
    w.show()
    sys.exit(app.exec())
