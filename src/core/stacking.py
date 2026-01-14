import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, SimilarityTransform

class Stacker:
    def __init__(self):
        pass

    def align_frames(self, frames, reference_frame=None):
        """
        Aligns a list of frames to a reference frame.
        If reference_frame is None, the first frame is used.
        Returns the aligned frames.
        """
        if not frames:
            return []
            
        if reference_frame is None:
            reference_frame = frames[0]
            
        aligned_frames = []
        
        # Ensure reference is grayscale for alignment calculation
        ref_gray = reference_frame
        if len(reference_frame.shape) == 3:
            ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
            
        for i, frame in enumerate(frames):
            if i == 0 and reference_frame is frames[0]:
                aligned_frames.append(frame)
                continue
                
            frame_gray = frame
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            # Calculate shift
            shift, error, diffphase = phase_cross_correlation(ref_gray, frame_gray, upsample_factor=10)
            
            # Apply shift
            # Shift is (y, x)
            # We can use warp or just roll if pixel integer shift, but warp is subpixel accurate
            # However, for speed and simplicity in initial version, we might stick to integer or warp
            
            # Create affine transform matrix
            rows, cols = frame_gray.shape
            M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
            
            if len(frame.shape) == 3:
                aligned_frame = cv2.warpAffine(frame, M, (cols, rows))
            else:
                aligned_frame = cv2.warpAffine(frame, M, (cols, rows))
                
            aligned_frames.append(aligned_frame)
            
        return aligned_frames

    def stack_frames(self, frames, method='mean'):
        """
        Stacks aligned frames using mean or median.
        """
        if not frames:
            return None
        
        # Validate all frames have the same shape
        first_shape = frames[0].shape
        for i, frame in enumerate(frames[1:], 1):
            if frame.shape != first_shape:
                raise ValueError(
                    f"Processing error: images must be same shape. "
                    f"Frame 0 has shape {first_shape}, but frame {i} has shape {frame.shape}. "
                    f"This typically happens when cropped frames have inconsistent dimensions."
                )
            
        stack = np.array(frames, dtype=np.float32)
        
        if method == 'median':
            result = np.median(stack, axis=0)
        else:
            result = np.mean(stack, axis=0)
            
        return np.clip(result, 0, 255).astype(np.uint8)
