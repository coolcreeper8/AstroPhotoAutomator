"""
Dual-object extraction and blending for fields containing both a planet and the Moon.

When the Moon and a planet share the same field of view (e.g. conjunctions), a single
stacking run must compromise: Moon-optimal settings (50% frame selection, aggressive
multi-scale sharpening) destroy fine planetary detail, while planet-optimal settings
(10-20% frame selection, optical flow, heavy L1-L2 sharpening) leave the Moon soft.

This module:
  1. Detects and separates the two objects using binary masks.
  2. Stacks each independently with its own optimal parameters (by re-using the
     existing Stacker pipeline via callback).
  3. Composites the results: Moon regions come from the Moon stack, planet regions
     from the planet stack, with a soft Gaussian-blended boundary between the two.

Usage
-----
The composite() function is the public entry point.  It accepts the raw stacked frame
(already processed by the caller's main pipeline) plus an independently produced
planet-optimised stack.  The caller is responsible for running the two separate stacks
and passing both images here.

blend_dual_object(moon_stack, planet_stack, planet_mask, blend_radius=30)
    Low-level blend: combine two equal-size arrays using a soft mask around the planet.

extract_object_mask(image, min_area_fraction=0.05)
    Locate the planet (small, bright) in a field that also contains the Moon.

composite(moon_stack, planet_stack, blend_radius=30)
    High-level entry: auto-detect planet location, build mask, blend, return result.
"""

import cv2
import numpy as np


def extract_object_mask(image, min_area_fraction=0.01, max_area_fraction=0.5):
    """
    Find the smaller, brighter object (planet) in a frame that also contains the Moon.

    Strategy:
      1. Convert to grayscale and apply Otsu threshold.
      2. Find all connected components (contours).
      3. The planet is the component whose area fraction sits between
         min_area_fraction and max_area_fraction of the total frame area.
         (The Moon fills >50 % of the frame in a close conjunction; the planet is
         a small disk.)
      4. Return a binary mask (uint8, 255 = planet region, 0 = background/Moon).

    Returns None if no suitable candidate is found.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = image.shape[0] * image.shape[1]
    planet_contour = None

    # Sort smallest first; the planet should be the small, isolated bright blob
    contours = sorted(contours, key=cv2.contourArea)
    for c in contours:
        area = cv2.contourArea(c)
        frac = area / frame_area
        if min_area_fraction <= frac <= max_area_fraction:
            planet_contour = c
            break

    if planet_contour is None:
        return None

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [planet_contour], -1, 255, -1)  # filled
    return mask


def blend_dual_object(moon_stack, planet_stack, planet_mask, blend_radius=30):
    """
    Composite moon_stack and planet_stack.

    The planet region (planet_mask == 255) is taken from planet_stack; everything
    else comes from moon_stack.  A Gaussian-blurred version of the mask produces
    a smooth alpha channel so there is no hard seam at the boundary.

    Parameters
    ----------
    moon_stack    : np.ndarray (H, W[, 3]) uint8  — Moon-optimised stack
    planet_stack  : np.ndarray (H, W[, 3]) uint8  — Planet-optimised stack
    planet_mask   : np.ndarray (H, W) uint8       — binary mask, 255 = planet
    blend_radius  : int  — sigma for Gaussian mask blur (controls transition width)

    Returns
    -------
    np.ndarray (H, W[, 3]) uint8 — composited image
    """
    if moon_stack.shape != planet_stack.shape:
        raise ValueError(
            f"Stack shapes must match for blending: {moon_stack.shape} vs {planet_stack.shape}"
        )

    # Build a soft float alpha mask: 1.0 = full planet stack, 0.0 = full moon stack
    alpha = planet_mask.astype(np.float32) / 255.0
    # Dilate slightly so the planet's sharpened halo is retained
    kernel = np.ones((5, 5), np.uint8)
    alpha_dilated = cv2.dilate((alpha * 255).astype(np.uint8), kernel, iterations=2)
    alpha = alpha_dilated.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), blend_radius)

    # Expand alpha to match image channels
    if len(moon_stack.shape) == 3:
        alpha = alpha[:, :, np.newaxis]

    m = moon_stack.astype(np.float32)
    p = planet_stack.astype(np.float32)
    blended = m * (1.0 - alpha) + p * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)


def composite(moon_stack, planet_stack, blend_radius=30):
    """
    High-level entry point: auto-locate the planet in planet_stack, build a mask,
    and blend moon_stack (background) with planet_stack (planet region).

    If no planet can be isolated, returns moon_stack unchanged and logs a warning.

    Parameters
    ----------
    moon_stack   : np.ndarray — Moon-optimised stack (used as base layer)
    planet_stack : np.ndarray — Planet-optimised stack (overlaid on detected planet region)
    blend_radius : int        — Gaussian sigma for the mask transition

    Returns
    -------
    np.ndarray — composited image, same shape as moon_stack
    """
    if moon_stack.shape != planet_stack.shape:
        # Resize planet_stack to moon_stack if they differ (shouldn't happen in normal use)
        h, w = moon_stack.shape[:2]
        planet_stack = cv2.resize(planet_stack, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = extract_object_mask(planet_stack)
    if mask is None:
        print("[dual_object] Warning: could not isolate planet region; returning Moon stack unchanged.")
        return moon_stack

    return blend_dual_object(moon_stack, planet_stack, mask, blend_radius=blend_radius)
