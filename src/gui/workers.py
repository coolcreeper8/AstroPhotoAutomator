from PyQt6.QtCore import QThread, pyqtSignal
import os
from core.video_loader import VideoLoader
from core.processing import FrameAnalyzer
from core.stacking import Stacker
from core.post_processing import WaveletEnhancer, ColorCorrector, AutoEnhancer
from core.derotation import derotate_frames, needs_derotation
from core.dual_object import composite, apply_exposure_boost, auto_exposure_factor
from core.planet_configs import get_config
import numpy as np
import cv2

class StackingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, video_paths, stack_val, stack_mode="percent", max_frames_load=None,
                 align_mode="translate", pano_mode=False, planet_config=None,
                 derotate=False, planet_name=None):
        super().__init__()
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]
        self.stack_val = stack_val
        self.stack_mode = stack_mode
        self.max_frames_load = max_frames_load
        self.align_mode = align_mode
        self.pano_mode = pano_mode
        self.planet_config = planet_config
        self.derotate = derotate
        self.planet_name = planet_name
        self.aborted = False

    def run(self):
        try:
            if not self.video_paths:
                self.error.emit("No videos to process.")
                return

            if self.pano_mode:
                stacked_parts = []
                for idx, filepath in enumerate(self.video_paths):
                    self.progress.emit(f"Pano: Loading video {idx+1}/{len(self.video_paths)} for tile...")
                    loader = VideoLoader(filepath)
                    tile_fps = loader.get_fps() or 25.0
                    frames = loader.load_all_frames(max_frames=self.max_frames_load)
                    loader.release()
                    if not frames:
                        continue
                    stacked_tile = self.process_single_stack(frames, fps=tile_fps, prefix=f"Tile {idx+1}")
                    if stacked_tile is not None:
                        stacked_parts.append(stacked_tile)

                if len(stacked_parts) > 1:
                    self.progress.emit("Stitching tiles into panorama (this may take a while)...")
                    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
                    status, pano = stitcher.stitch(stacked_parts)

                    if status != cv2.Stitcher_OK:
                        self.error.emit(f"Panorama stitching failed (error code {status}). Returning first tile instead.")
                        self.finished.emit(stacked_parts[0])
                        return
                    self.finished.emit(pano)
                elif len(stacked_parts) == 1:
                    self.finished.emit(stacked_parts[0])
                else:
                    self.error.emit("No tiles successfully stacked.")
                return

            # --- ALL-IN-ONE Stack ---
            all_frames = []
            combined_fps = 25.0
            for idx, filepath in enumerate(self.video_paths):
                self.progress.emit(f"Loading video {idx+1}/{len(self.video_paths)}: {os.path.basename(filepath)}")
                loader = VideoLoader(filepath)
                if idx == 0:
                    combined_fps = loader.get_fps() or 25.0
                frames = loader.load_all_frames(max_frames=self.max_frames_load)
                loader.release()

                if frames:
                    all_frames.extend(frames)
                    self.progress.emit(f"  Loaded {len(frames)} frames from video {idx+1}")

            if not all_frames:
                self.error.emit("No frames loaded from any video.")
                return

            self.progress.emit(f"Total: {len(all_frames)} frames loaded from {len(self.video_paths)} video(s).")
            stacked_image = self.process_single_stack(all_frames, fps=combined_fps)
            if stacked_image is not None:
                self.finished.emit(stacked_image)
            
        except Exception as e:
            self.error.emit(str(e))

    def process_single_stack(self, frames, fps=25.0, prefix=""):
        pfx = f"{prefix}: " if prefix else ""
        self.progress.emit(f"{pfx}Analyzing quality...")

        # 1. Analyze Quality & Center; track each frame's original position for derotation
        analyzed_frames = []
        orig_frame_indices = []
        qualities = []

        h, w = frames[0].shape[:2]
        crop_size = (min(w, h), min(w, h))

        for i, frame in enumerate(frames):
            if self.isInterruptionRequested():
                return None

            if i % 50 == 0:
                self.progress.emit(f"{pfx}Analyzing frame {i+1}/{len(frames)}...")

            center = FrameAnalyzer.detect_roi(frame)
            if center is None:
                continue

            cropped = FrameAnalyzer.crop_centered(frame, center, crop_size)
            q = FrameAnalyzer.estimate_quality(cropped)

            analyzed_frames.append(cropped)
            orig_frame_indices.append(i)
            qualities.append(q)

        # 2. Sort by Quality
        self.progress.emit(f"{pfx}Sorting frames by quality...")
        sorted_indices = np.argsort(qualities)[::-1]

        if self.stack_mode == "percent":
            num_to_stack = max(1, int(len(frames) * (self.stack_val / 100.0)))
        elif self.stack_mode == "auto":
            self.progress.emit(f"{pfx}Analyzing quality graph to find optimal stack size...")
            best_percent = FrameAnalyzer.analyze_quality_graph(qualities)
            num_to_stack = max(1, int(len(frames) * (best_percent / 100.0)))
            self.progress.emit(f"{pfx}Auto-Optimizer selected: {best_percent}% ({num_to_stack} frames)")
        else:  # Count
            num_to_stack = min(len(analyzed_frames), self.stack_val)

        best_indices = sorted_indices[:num_to_stack]
        best_frames = [analyzed_frames[i] for i in best_indices]
        best_orig_indices = [orig_frame_indices[i] for i in best_indices]

        # 3. Planetary derotation (before alignment so the reference is at t=0 orientation)
        if self.derotate and self.planet_name and needs_derotation(self.planet_name):
            self.progress.emit(
                f"{pfx}Derotating {len(best_frames)} frames for {self.planet_name} "
                f"(fps={fps:.1f}, period={self.planet_name})..."
            )
            best_frames = derotate_frames(best_frames, best_orig_indices, fps, self.planet_name)

        # 4. Create Reference Stack
        self.progress.emit(f"{pfx}Creating reference stack...")
        stacker = Stacker()

        ref_count = max(5, len(best_frames) // 2)
        reference_frames = best_frames[:ref_count]
        reference_stack = stacker.stack_frames(reference_frames, method='mean')

        # 5. Align to Reference Stack
        of_params    = self.planet_config.get("of_params")    if self.planet_config else None
        ecc_criteria = self.planet_config.get("ecc_criteria") if self.planet_config else None
        self.progress.emit(f"{pfx}Aligning {len(best_frames)} frames to reference ({self.align_mode})...")
        aligned_frames = stacker.align_frames(
            best_frames, reference_frame=reference_stack, mode=self.align_mode,
            of_params=of_params, ecc_criteria=ecc_criteria
        )

        # 6. Final Stack
        self.progress.emit(f"{pfx}Creating final stack...")
        stacked_image = stacker.stack_frames(aligned_frames, method='mean')

        return stacked_image


class PostProcessingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, stacked_image, wavelet_layers, auto_color, denoise=0, auto_mode=False,
                 planet_config=None):
        super().__init__()
        self.stacked_image = stacked_image
        self.wavelet_layers = wavelet_layers
        self.auto_color = auto_color
        self.denoise = denoise
        self.auto_mode = auto_mode
        self.planet_config = planet_config

    def run(self):
        try:
            result = self.stacked_image.copy()

            if self.auto_mode:
                self.progress.emit("Running smart automatic optimization...")
                result = AutoEnhancer.optimize(result, planet_config=self.planet_config)
            else:
                if self.wavelet_layers:
                    self.progress.emit(f"Applying wavelet sharpening (Denoise: {self.denoise})...")
                    result = WaveletEnhancer.apply_wavelets(result, self.wavelet_layers, denoise_strength=self.denoise)
                    
                if self.auto_color:
                    self.progress.emit("Applying color correction...")
                    result = ColorCorrector.align_channels(result)
                    result = ColorCorrector.auto_balance(result)
                
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class DualObjectWorker(QThread):
    """
    Produces both a Moon-optimised stack and a planet-optimised stack from the same
    source video, then composites the planet region over the Moon background.

    Key problem: planetary video is captured at short exposure tuned to the bright
    planet.  The Moon — though physically far brighter — is underexposed in those
    frames (sometimes nearly invisible for Jupiter/Saturn conjunctions).
    This worker applies apply_exposure_boost() to all frames before the Moon stacking
    pass so the Moon surface is recovered; the planet pass uses the raw frames.

    moon_boost_factor: linear brightness multiplier for the Moon pass.
        0.0  = auto-detect from the first frame via auto_exposure_factor().
        >0.0 = use the given value directly (user override).
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, video_paths, max_frames_load=None,
                 planet_name="Jupiter", moon_boost_factor=0.0, blend_radius=30):
        super().__init__()
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]
        self.max_frames_load = max_frames_load
        self.planet_name = planet_name
        self.moon_boost_factor = moon_boost_factor
        self.blend_radius = blend_radius

    def run(self):
        try:
            # 1. Load all frames once — both passes will operate on this list
            self.progress.emit("Dual-blend: loading frames...")
            all_frames = []
            fps = 25.0
            for idx, filepath in enumerate(self.video_paths):
                loader = VideoLoader(filepath)
                if idx == 0:
                    fps = loader.get_fps() or 25.0
                frames = loader.load_all_frames(max_frames=self.max_frames_load)
                loader.release()
                all_frames.extend(frames)

            if not all_frames:
                self.error.emit("Dual-blend: no frames could be loaded.")
                return

            # 2. Determine exposure boost factor
            if self.moon_boost_factor > 0.0:
                boost = self.moon_boost_factor
                self.progress.emit(f"Dual-blend: Moon exposure boost = {boost:.1f}x (manual).")
            else:
                boost = auto_exposure_factor(all_frames[0])
                self.progress.emit(
                    f"Dual-blend: auto Moon exposure boost = {boost:.1f}x "
                    f"(estimated from first frame)."
                )

            # 3. Moon pass — boost frames so the dim Moon surface becomes visible,
            #    then stack with Moon-optimal config (50%, optical flow, wide window)
            self.progress.emit(f"Dual-blend: brightening {len(all_frames)} frames for Moon stack...")
            moon_frames = [apply_exposure_boost(f, factor=boost) for f in all_frames]

            moon_cfg = get_config("Moon (Surface)")
            moon_worker = StackingWorker(
                self.video_paths,
                stack_val=moon_cfg["stack_percent"],
                stack_mode="percent",
                max_frames_load=self.max_frames_load,
                align_mode=moon_cfg["align_mode"],
                planet_config=moon_cfg,
            )
            self.progress.emit(
                f"Dual-blend: stacking Moon ({moon_cfg['stack_percent']}% frames, "
                f"{moon_cfg['align_mode']})..."
            )
            moon_stack = moon_worker.process_single_stack(moon_frames, fps=fps, prefix="Moon")
            if moon_stack is None:
                self.error.emit("Dual-blend: Moon stack failed.")
                return
            moon_stack = WaveletEnhancer.apply_wavelets(
                moon_stack,
                layers=moon_cfg["wavelet_layers"],
                denoise_strength=moon_cfg["denoise"],
            )

            # 4. Planet pass — original frames (correct planet exposure), planet-optimal config
            planet_cfg = get_config(self.planet_name)
            planet_worker = StackingWorker(
                self.video_paths,
                stack_val=planet_cfg["stack_percent"],
                stack_mode="percent",
                max_frames_load=self.max_frames_load,
                align_mode=planet_cfg["align_mode"],
                planet_config=planet_cfg,
            )
            self.progress.emit(
                f"Dual-blend: stacking planet ({planet_cfg['stack_percent']}% frames, "
                f"{planet_cfg['align_mode']})..."
            )
            planet_stack = planet_worker.process_single_stack(all_frames, fps=fps, prefix="Planet")
            if planet_stack is None:
                self.error.emit("Dual-blend: planet stack failed.")
                return
            planet_stack = WaveletEnhancer.apply_wavelets(
                planet_stack,
                layers=planet_cfg["wavelet_layers"],
                denoise_strength=planet_cfg["denoise"],
            )

            # 5. Composite: Moon stack as base, planet stack overlaid on the detected planet region
            self.progress.emit("Dual-blend: compositing Moon and planet stacks...")
            result = composite(moon_stack, planet_stack, blend_radius=self.blend_radius)

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
