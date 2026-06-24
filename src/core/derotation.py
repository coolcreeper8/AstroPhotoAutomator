"""
Planetary derotation for fast-rotating targets.

During a typical 10-minute Jupiter capture, the equatorial zone drifts ~1° —
enough to smear fine belt/zone structure when stacking without compensation.
This module computes per-frame counter-rotation angles based on elapsed capture
time and the planet's known sidereal rotation period, then applies them before
alignment and stacking.

The reference orientation is always frame 0 of the selected set.  Each subsequent
frame is rotated back by -(elapsed_time / period) * 360° around the detected
planet centre so all frames share the same rotational phase before being stacked.

Supported targets and their equatorial (System I) periods:
  Jupiter: 9.925 h  (~1.0°/min at the equator — derotation matters above ~5 min)
  Saturn:  10.66 h  (~0.94°/min — noticeable in very long sessions)

All other planets rotate slowly enough that derotation during typical imaging
windows is negligible (Mars: ~0.25°/10 min).
"""

import cv2
import numpy as np

# Sidereal rotation periods in seconds (equatorial / System I where applicable)
ROTATION_PERIODS_S = {
    "Jupiter":         9.925  * 3600,
    "Jupiter / Venus": 9.925  * 3600,   # Ambiguous detection → use Jupiter
    "Saturn":          10.656 * 3600,
}


def needs_derotation(planet_name):
    """Return True if this planet benefits from derotation."""
    return planet_name in ROTATION_PERIODS_S


def derotate_frames(frames, original_indices, fps, planet_name):
    """
    Counter-rotate each frame around the planet centre by the angle the planet
    has rotated since the first selected frame.

    Parameters
    ----------
    frames         : list of np.ndarray — cropped, quality-selected frames (any order)
    original_indices : list of int — position of each frame in the original loaded sequence
    fps            : float — frames per second of the source video
    planet_name    : str  — key into ROTATION_PERIODS_S

    Returns
    -------
    list of np.ndarray — same length as `frames`, each rotated to the t=0 orientation
    """
    period = ROTATION_PERIODS_S.get(planet_name)
    if period is None or fps <= 0 or not frames:
        return frames

    # Detect planet centre from the first frame in the selection (lowest original index)
    # This becomes the fixed pivot point for all counter-rotations.
    anchor_idx = min(range(len(frames)), key=lambda k: original_indices[k])
    anchor_frame = frames[anchor_idx]
    h, w = anchor_frame.shape[:2]

    # Find planet centre via brightness centre-of-mass
    if len(anchor_frame.shape) == 3:
        gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = anchor_frame

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = w // 2, h // 2

    center = (float(cx), float(cy))

    # Reference time = earliest frame in the selection
    t0 = min(original_indices) / fps

    derotated = []
    for frame, orig_idx in zip(frames, original_indices):
        elapsed = orig_idx / fps - t0
        # Planet has rotated forward; we rotate the frame backward to compensate.
        # Angle is negative because cv2 uses CCW-positive convention, and we want
        # to undo the forward (CW in sky-north-up view) drift.
        angle_deg = -(elapsed / period) * 360.0

        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        derotated.append(rotated)

    return derotated
