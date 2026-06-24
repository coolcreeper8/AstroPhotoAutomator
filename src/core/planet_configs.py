"""
Planet-specific processing profiles informed by AutoStakkert!4 and Registax 6 workflows.

Wavelet weights at dyadic scales (1,2,4,8,16,32 px) follow Registax 6 planetary guides:
  Jupiter  — heavy L1-L2 (belt/zone detail ~1-3 arcsec), L3 for GRS/festoon scale
  Saturn   — balanced L1-L3; Cassini division maps to ~2-4 px at f/20
  Moon     — aggressive across all scales (rilles to maria boundaries)
  Mars     — moderate with higher denoise (small disk = low SNR)

Optical flow params (Farneback): winsize and levels tuned to object angular size and
typical atmospheric coherence length.  Wider objects under stronger seeing need larger windows.

ECC criteria (max_iter, epsilon): Saturn uses 100 iterations for precise ring registration.
"""

# Farneback optical flow parameter sets
_OF_JUPITER = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2)
_OF_MOON    = dict(pyr_scale=0.5, levels=5, winsize=21, iterations=5, poly_n=7, poly_sigma=1.5)
_OF_MARS    = dict(pyr_scale=0.5, levels=2, winsize=11, iterations=3, poly_n=5, poly_sigma=1.1)
_OF_DEFAULT = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2)

# ECC termination criteria as (max_iterations, epsilon) — converted in stacking.py
_ECC_PRECISE = (100, 0.0005)   # Saturn rings: wide span, needs precise angular registration
_ECC_DEFAULT = (50,  0.001)

PLANET_CONFIGS = {
    "Jupiter": {
        "stack_percent":      15,
        "align_mode":         "optical_flow",
        "of_params":          _OF_JUPITER,
        "ecc_criteria":       _ECC_DEFAULT,
        # Heavy L1-L2: fine belt/zone boundary detail; L3 for GRS/festoon scale features
        "wavelet_layers":     [(1.0, 1.8), (2.0, 1.5), (4.0, 1.0), (8.0, 0.5), (16.0, 0.2), (32.0, 0.0)],
        "denoise":            3,
        # Standard erosion keeps ringing away from limb
        "limb_erosion_iters": 2,
        "description": (
            "15% frame selection (9.9 h rotation causes smear above ~25%); "
            "optical flow corrects local seeing across belt/zone structure; "
            "heavy L1-L2 wavelet for fine detail per Registax planetary workflow."
        ),
    },
    "Jupiter / Venus": {
        # Ambiguous detection — default to Jupiter since Venus has no visible surface detail
        "stack_percent":      15,
        "align_mode":         "optical_flow",
        "of_params":          _OF_JUPITER,
        "ecc_criteria":       _ECC_DEFAULT,
        "wavelet_layers":     [(1.0, 1.8), (2.0, 1.5), (4.0, 1.0), (8.0, 0.5), (16.0, 0.2), (32.0, 0.0)],
        "denoise":            3,
        "limb_erosion_iters": 2,
        "description": "Ambiguous detection — Jupiter preset applied as conservative default.",
    },
    "Saturn": {
        "stack_percent":      20,
        # Affine corrects ring-plane tilt drift that accumulates during long capture sessions
        "align_mode":         "affine",
        "of_params":          _OF_DEFAULT,
        "ecc_criteria":       _ECC_PRECISE,
        # Balanced L1-L3: Cassini division at ~2-4 px, ring banding mid-scale, disk belts coarse
        "wavelet_layers":     [(1.0, 1.2), (2.0, 1.5), (4.0, 1.2), (8.0, 0.8), (16.0, 0.3), (32.0, 0.1)],
        "denoise":            5,
        # Rings extend past disk — extra erosion prevents ringing artifacts at ring tips
        "limb_erosion_iters": 3,
        "description": (
            "20% frames; affine corrects ring-plane tilt drift; "
            "100-iteration ECC for precise ring registration; "
            "balanced L1-L3 covers Cassini division (~2-4 px) through disk banding."
        ),
    },
    "Mars": {
        "stack_percent":      30,
        # Small disk: translation sufficient; affine/flow adds noise without resolution gain
        "align_mode":         "translate",
        "of_params":          _OF_MARS,
        "ecc_criteria":       _ECC_DEFAULT,
        # Moderate L2-L3: albedo features (Syrtis Major, polar caps) are mid-scale
        "wavelet_layers":     [(1.0, 1.0), (2.0, 1.3), (4.0, 1.0), (8.0, 0.5), (16.0, 0.2), (32.0, 0.0)],
        "denoise":            7,
        "limb_erosion_iters": 2,
        "description": (
            "30% frames (slow 24.6 h rotation allows more usable frames); "
            "translation sufficient for small disk; "
            "higher denoise compensates for low-SNR small apparent disk."
        ),
    },
    "Moon (Surface)": {
        "stack_percent":      50,
        # Wide field — different regions shift independently under atmospheric seeing
        "align_mode":         "optical_flow",
        # Larger window (21 px) and more pyramid levels for wide-field distortion correction
        "of_params":          _OF_MOON,
        "ecc_criteria":       _ECC_DEFAULT,
        # Aggressive multi-scale: L1-L2 for fine rilles, L3-L4 for crater walls,
        # L5-L6 for large maria/highland boundaries
        "wavelet_layers":     [(1.0, 2.0), (2.0, 2.0), (4.0, 1.5), (8.0, 1.0), (16.0, 0.5), (32.0, 0.3)],
        "denoise":            2,
        # Moon fills frame; minimal erosion since limb is at frame edge not in center
        "limb_erosion_iters": 1,
        "description": (
            "50% frames; optical flow with 21-px window for wide-field distortion; "
            "aggressive multi-scale: L1-L2 for rilles, L3-L4 for crater walls, "
            "L5-L6 for maria/highland boundaries; minimal denoise to preserve fine detail."
        ),
    },
    "Planet (Jupiter/Mars/Venus)": {
        "stack_percent":      20,
        "align_mode":         "translate",
        "of_params":          _OF_DEFAULT,
        "ecc_criteria":       _ECC_DEFAULT,
        "wavelet_layers":     [(1.0, 0.0), (2.0, 1.0), (4.0, 1.0), (8.0, 0.5), (16.0, 0.0), (32.0, 0.0)],
        "denoise":            5,
        "limb_erosion_iters": 2,
        "description": "Generic planet: conservative defaults.",
    },
    "Unknown Celestial Body": {
        "stack_percent":      20,
        "align_mode":         "translate",
        "of_params":          _OF_DEFAULT,
        "ecc_criteria":       _ECC_DEFAULT,
        "wavelet_layers":     [(1.0, 0.0), (2.0, 1.0), (4.0, 1.0), (8.0, 0.5), (16.0, 0.0), (32.0, 0.0)],
        "denoise":            5,
        "limb_erosion_iters": 2,
        "description": "Unknown object: conservative defaults.",
    },
}

DEFAULT_CONFIG = PLANET_CONFIGS["Unknown Celestial Body"]

# Maps GUI target selector labels -> PLANET_CONFIGS keys (None = auto-detect after stacking)
TARGET_LABEL_TO_KEY = {
    "Auto-Detect":      None,
    "Jupiter":          "Jupiter",
    "Saturn":           "Saturn",
    "Mars":             "Mars",
    "Moon (Surface)":   "Moon (Surface)",
    "Planet (Generic)": "Planet (Jupiter/Mars/Venus)",
}


def get_config(object_name):
    """Return the planet config for a recognized object name, or DEFAULT_CONFIG."""
    return PLANET_CONFIGS.get(object_name, DEFAULT_CONFIG)
