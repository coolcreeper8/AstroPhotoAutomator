from PyQt6.QtCore import QThread, pyqtSignal
from core.video_loader import VideoLoader
from core.processing import FrameAnalyzer
from core.stacking import Stacker
from core.post_processing import WaveletEnhancer, ColorCorrector, AutoEnhancer
import numpy as np

class StackingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, video_paths, stack_val, stack_mode="percent", max_frames_load=None, align_mode="translate"):
        super().__init__()
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]
        # In 'auto' mode, stack_val is ignored or used as fallback/min
        self.stack_val = stack_val 
        self.stack_mode = stack_mode
        self.max_frames_load = max_frames_load
        self.align_mode = align_mode
        self.aborted = False

    def run(self):
        try:
            # Load frames from all videos
            all_frames = []
            
            for idx, filepath in enumerate(self.video_paths):
                self.progress.emit(f"Loading video {idx+1}/{len(self.video_paths)}: {filepath.split('/')[-1]}")
                loader = VideoLoader(filepath)
                frames = loader.load_all_frames(max_frames=self.max_frames_load)
                loader.release()
                
                if frames:
                    all_frames.extend(frames)
                    self.progress.emit(f"  Loaded {len(frames)} frames from video {idx+1}")
            
            if not all_frames:
                self.error.emit("No frames loaded from any video.")
                return

            self.progress.emit(f"Total: {len(all_frames)} frames loaded from {len(self.video_paths)} video(s). Analyzing quality...")
            
            # 1. Analyze Quality & Center
            analyzed_frames = []
            qualities = []
            
            # First pass: Center and Crop
            h, w = all_frames[0].shape[:2]
            crop_size = (min(w, h), min(w, h))

            for i, frame in enumerate(all_frames):
                if self.isInterruptionRequested(): return
                
                if i % 50 == 0:
                    self.progress.emit(f"Analyzing frame {i+1}/{len(all_frames)}...")
                
                center = FrameAnalyzer.detect_roi(frame)
                cropped = FrameAnalyzer.crop_centered(frame, center, crop_size)
                q = FrameAnalyzer.estimate_quality(cropped)
                
                analyzed_frames.append(cropped)
                qualities.append(q)
                
            # 2. Sort by Quality
            self.progress.emit("Sorting frames by quality...")
            sorted_indices = np.argsort(qualities)[::-1]
            sorted_qualities = np.sort(qualities)[::-1] # Use unsorted originally? No, need sorted for graph
            
            if self.stack_mode == "percent":
                num_to_stack = max(1, int(len(all_frames) * (self.stack_val / 100.0)))
            elif self.stack_mode == "auto":
                self.progress.emit("Analyzing quality graph to find optimal stack size...")
                best_percent = FrameAnalyzer.analyze_quality_graph(qualities) # Pass original unsorted/list
                num_to_stack = max(1, int(len(all_frames) * (best_percent / 100.0)))
                self.progress.emit(f"Auto-Optimizer selected: {best_percent}% ({num_to_stack} frames)")
            else: # Count
                num_to_stack = min(len(all_frames), self.stack_val)
                
            best_indices = sorted_indices[:num_to_stack]
            
            best_frames = [analyzed_frames[i] for i in best_indices]
            
            # 3. Create Reference Stack (AutoStakkert 3 approach)
            self.progress.emit("Creating reference stack...")
            stacker = Stacker()
            
            # Use top 50% of best frames to create reference, or at least 5
            ref_count = max(5, len(best_frames) // 2)
            reference_frames = best_frames[:ref_count]
            reference_stack = stacker.stack_frames(reference_frames, method='mean')
            
            # 4. Align to Reference Stack
            self.progress.emit(f"Aligning {len(best_frames)} frames to reference ({self.align_mode})...")
            aligned_frames = stacker.align_frames(best_frames, reference_frame=reference_stack, mode=self.align_mode)
            
            # 5. Final Stack
            self.progress.emit("Creating final stack...")
            stacked_image = stacker.stack_frames(aligned_frames, method='mean')
            
            self.finished.emit(stacked_image)
            
        except Exception as e:
            self.error.emit(str(e))


class PostProcessingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, stacked_image, wavelet_layers, auto_color, denoise=0, auto_mode=False):
        super().__init__()
        self.stacked_image = stacked_image
        self.wavelet_layers = wavelet_layers
        self.auto_color = auto_color
        self.denoise = denoise
        self.auto_mode = auto_mode

    def run(self):
        try:
            result = self.stacked_image.copy()
            
            if self.auto_mode:
                self.progress.emit("Running smart automatic optimization...")
                result = AutoEnhancer.optimize(result)
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
