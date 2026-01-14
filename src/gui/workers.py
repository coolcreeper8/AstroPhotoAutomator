from PyQt6.QtCore import QThread, pyqtSignal
from core.video_loader import VideoLoader
from core.processing import FrameAnalyzer
from core.stacking import Stacker
from core.post_processing import WaveletEnhancer, ColorCorrector
import numpy as np

class StackingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, video_paths, stack_percent):
        super().__init__()
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]
        self.stack_percent = stack_percent
        self.aborted = False

    def run(self):
        try:
            # Load frames from all videos
            all_frames = []
            
            for idx, filepath in enumerate(self.video_paths):
                self.progress.emit(f"Loading video {idx+1}/{len(self.video_paths)}: {filepath.split('/')[-1]}")
                loader = VideoLoader(filepath)
                frames = loader.load_all_frames(max_frames=1000)
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
            num_to_stack = max(1, int(len(all_frames) * (self.stack_percent / 100.0)))
            best_indices = sorted_indices[:num_to_stack]
            
            best_frames = [analyzed_frames[i] for i in best_indices]
            
            # 3. Create Reference Stack (AutoStakkert 3 approach)
            self.progress.emit("Creating reference stack...")
            stacker = Stacker()
            
            # Use top 50% of best frames to create reference
            ref_count = max(5, len(best_frames) // 2)
            reference_frames = best_frames[:ref_count]
            reference_stack = stacker.stack_frames(reference_frames, method='mean')
            
            # 4. Align to Reference Stack
            self.progress.emit(f"Aligning {len(best_frames)} frames to reference...")
            aligned_frames = stacker.align_frames(best_frames, reference_frame=reference_stack)
            
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

    def __init__(self, stacked_image, wavelet_layers, auto_color):
        super().__init__()
        self.stacked_image = stacked_image
        self.wavelet_layers = wavelet_layers
        self.auto_color = auto_color

    def run(self):
        try:
            result = self.stacked_image.copy()
            
            if self.wavelet_layers:
                self.progress.emit("Applying wavelet sharpening...")
                result = WaveletEnhancer.apply_wavelets(result, self.wavelet_layers)
                
            if self.auto_color:
                self.progress.emit("Applying color correction...")
                result = ColorCorrector.align_channels(result)
                result = ColorCorrector.auto_balance(result)
                
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
