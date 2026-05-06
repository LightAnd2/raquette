"""
Court homography — maps pixel coordinates to normalised court coordinates.

A tennis court is 23.77m × 10.97m (singles). We normalise to [0,1] × [0,1]
so heatmaps are resolution-independent.

Usage:
    court = CourtMapper()
    court.calibrate(frame)          # interactive: click 4 corners in order
    norm_xy = court.to_court(px, py)
    px, py  = court.to_pixel(norm_x, norm_y)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


# Standard court corners in metres (top-left baseline, top-right, bottom-right, bottom-left)
COURT_REAL_CORNERS = np.array([
    [0.0,   0.0],
    [10.97, 0.0],
    [10.97, 23.77],
    [0.0,   23.77],
], dtype=np.float32)


class CourtMapper:
    """
    Perspective transform between pixel space and normalised [0,1]² court space.

    Two calibration modes:
    1. Manual: call calibrate(frame) to click 4 corners interactively
    2. Auto: pass pixel_corners directly if you have them from another source
    """

    def __init__(self):
        self._H: Optional[np.ndarray] = None   # pixel → normalised
        self._H_inv: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    def calibrate(self, frame: np.ndarray) -> None:
        """Click 4 court corners (TL, TR, BR, BL) to compute the homography."""
        points = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append([x, y])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Calibrate court — click TL, TR, BR, BL", frame)

        cv2.namedWindow("Calibrate court — click TL, TR, BR, BL")
        cv2.setMouseCallback("Calibrate court — click TL, TR, BR, BL", on_click)
        cv2.imshow("Calibrate court — click TL, TR, BR, BL", frame)

        while len(points) < 4:
            if cv2.waitKey(50) == 27:
                break

        cv2.destroyAllWindows()

        if len(points) == 4:
            self.set_pixel_corners(np.array(points, dtype=np.float32))

    def set_pixel_corners(self, pixel_corners: np.ndarray) -> None:
        """
        Set calibration from known pixel corners (TL, TR, BR, BL).
        pixel_corners: (4, 2) array of (x, y) pixel coordinates.
        """
        # Normalise real-world corners to [0,1]
        norm_corners = COURT_REAL_CORNERS / COURT_REAL_CORNERS.max()
        self._H, _ = cv2.findHomography(pixel_corners, norm_corners)
        self._H_inv, _ = cv2.findHomography(norm_corners, pixel_corners)

    def to_court(self, px: float, py: float) -> tuple[float, float]:
        """Pixel coords → normalised court coords."""
        if not self.is_calibrated:
            raise RuntimeError("CourtMapper not calibrated")
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        x, y = result[0][0]
        return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))

    def to_pixel(self, cx: float, cy: float) -> tuple[float, float]:
        """Normalised court coords → pixel coords."""
        if not self.is_calibrated:
            raise RuntimeError("CourtMapper not calibrated")
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H_inv)
        return float(result[0][0][0]), float(result[0][0][1])

    def save(self, path: str) -> None:
        np.save(path, self._H)

    def load(self, path: str) -> None:
        self._H = np.load(path)
        self._H_inv = np.linalg.inv(self._H)


def auto_detect_court(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to automatically detect court lines using edge detection + Hough.
    Returns (4, 2) pixel corner array or None if detection fails.

    This is a best-effort heuristic — for production, manual calibration gives
    more reliable results.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return None

    h, w = frame.shape[:2]

    # Rough heuristic: find the outermost lines in each quadrant
    top = min(lines, key=lambda l: min(l[0][1], l[0][3]))
    bottom = max(lines, key=lambda l: max(l[0][1], l[0][3]))
    left = min(lines, key=lambda l: min(l[0][0], l[0][2]))
    right = max(lines, key=lambda l: max(l[0][0], l[0][2]))

    corners = np.array([
        [left[0][0], top[0][1]],
        [right[0][0], top[0][1]],
        [right[0][2], bottom[0][3]],
        [left[0][2], bottom[0][3]],
    ], dtype=np.float32)

    return corners
