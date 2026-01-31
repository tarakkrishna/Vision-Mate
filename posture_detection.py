from __future__ import annotations

from typing import Optional, Tuple


class PostureDetector:
    """Lightweight heuristic posture monitor based on face position.

    This does not use heavy landmark models. Instead, it estimates posture
    from the vertical position of the face bounding box in the frame. When
    the face consistently sits much lower than the frame centre for several
    seconds, this is treated as a slouched posture and a gentle alert can
    be raised by the caller.
    """

    def __init__(
        self,
        tilt_threshold: float = 0.3,
        min_poor_duration: float = 3.0,
        alert_cooldown_seconds: float = 600.0,
    ) -> None:
        # How far below the frame centre (as a fraction of half-height)
        # counts as "poor" posture.
        self.tilt_threshold = float(tilt_threshold)
        # How long posture must be continuously poor before we alert.
        self.min_poor_duration = float(min_poor_duration)
        # Cooldown between posture alerts.
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)

        self.last_alert_time: float = 0.0
        self.poor_since: Optional[float] = None
        self.current_state: str = "unknown"

        # Rough accounting of how long posture has been poor over the session.
        self.poor_posture_seconds: float = 0.0
        self._last_update_time: Optional[float] = None
        self.alert_count: int = 0

    def reset(self) -> None:
        self.last_alert_time = 0.0
        self.poor_since = None
        self.current_state = "unknown"
        self.poor_posture_seconds = 0.0
        self._last_update_time = None
        self.alert_count = 0

    def update(self, face_rect: Tuple[int, int, int, int], frame_shape, now: float) -> dict:
        """Update posture estimate from the latest face rectangle.

        Parameters
        ----------
        face_rect: (x, y, w, h) of the detected face.
        frame_shape: tuple
            Shape of the grayscale or colour frame (H, W[, C]).
        now: float
            Current time.time() value.

        Returns
        -------
        dict with keys:
            - state: "good" or "poor"
            - posture_alert: bool
        """

        height = int(frame_shape[0])
        width = int(frame_shape[1])

        x, y, w, h = face_rect
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Normalised vertical offset from frame centre. Positive values mean
        # the face is lower in the frame (more likely slouched).
        norm_offset_y = (cy - height / 2.0) / (height / 2.0)
        poor = norm_offset_y > self.tilt_threshold

        # Time delta since last update, for rough integration of poor duration.
        dt = 0.0
        if self._last_update_time is not None:
            dt = max(0.0, now - self._last_update_time)
        self._last_update_time = now

        posture_alert = False

        if poor:
            if self.poor_since is None:
                self.poor_since = now
            else:
                # Accumulate time spent in poor posture.
                self.poor_posture_seconds += dt

                if now - self.poor_since >= self.min_poor_duration:
                    if now - self.last_alert_time >= self.alert_cooldown_seconds:
                        self.last_alert_time = now
                        posture_alert = True
                        self.alert_count += 1
        else:
            self.poor_since = None

        self.current_state = "poor" if poor else "good"

        return {
            "state": self.current_state,
            "posture_alert": posture_alert,
        }

    def session_stats(self) -> dict:
        """Return aggregate posture stats for the current monitoring session."""

        return {
            "poor_posture_seconds": round(self.poor_posture_seconds, 1),
            "poor_posture_alerts": int(self.alert_count),
        }
