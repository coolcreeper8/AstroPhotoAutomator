import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, SimilarityTransform

class Stacker:
    def __init__(self):
        pass

    def align_frames(self, frames, reference_frame=None, mode='translate'):
        """
        Aligns a list of frames to a reference frame.
        Modes:
        - 'translate': Simple shift (fast, handles tracking drift).
        - 'affine': shift + rotation (handles field rotation).
        - 'optical_flow': pixel-wise warping (handles atmospheric distortion/seeing).
        """
        if not frames:
            return []
            
        if reference_frame is None:
            reference_frame = frames[0]
            
        aligned_frames = []
        ref_gray = reference_frame
        
        if len(reference_frame.shape) == 3:
            ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
            
        # Sharpen reference frame for better alignment lock (Critical for AutoStakkert-like performance)
        # Unsharp Mask: Original = Original + Amount * (Original - Blurred)
        blur = cv2.GaussianBlur(ref_gray, (0, 0), 3.0)
        ref_gray = cv2.addWeighted(ref_gray, 2.0, blur, -1.0, 0)
            
        for i, frame in enumerate(frames):
            # If it's the reference frame itself (and first), just append (optional optimization)
            # But usually we align everything to the *computed* reference stack
            
            frame_gray = frame
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            aligned_frame = None
            
            try:
                if mode == 'optical_flow':
                    aligned_frame = self._align_optical_flow(ref_gray, frame_gray, frame)
                elif mode == 'affine':
                    aligned_frame = self._align_affine(ref_gray, frame_gray, frame)
                else: # translate
                    aligned_frame = self._align_translation(ref_gray, frame_gray, frame)
            except Exception as e:
                print(f"Alignment failed for frame {i}: {e}.Using original.")
                aligned_frame = frame

            aligned_frames.append(aligned_frame)
            
        return aligned_frames

    def _align_translation(self, ref_gray, frame_gray, frame_color):
        # Calculate shift
        shift, error, diffphase = phase_cross_correlation(ref_gray, frame_gray, upsample_factor=10)
        
        # Shift is (y, x)
        rows, cols = frame_gray.shape
        M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
        
        return cv2.warpAffine(frame_color, M, (cols, rows))

    def _align_affine(self, ref_gray, frame_gray, frame_color):
        # Initialize Warp Matrix (2x3 for Affine)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        
        try:
            # Use ECC (Enhanced Correlation Coefficient) - robust and accurate
            # Note: Input images must be single channel
            (cc, warp_matrix) = cv2.findTransformECC(ref_gray, frame_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
            
            rows, cols = frame_gray.shape
            aligned = cv2.warpAffine(frame_color, warp_matrix, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned
        except cv2.error:
            # Fallback to translation if affine fails
            return self._align_translation(ref_gray, frame_gray, frame_color)

    def _align_optical_flow(self, ref_gray, frame_gray, frame_color):
        # 1. First do a rough translation alignment to reduce displacement
        rough_aligned = self._align_translation(ref_gray, frame_gray, frame_color)
        rough_gray = rough_aligned
        if len(rough_aligned.shape) == 3:
            rough_gray = cv2.cvtColor(rough_aligned, cv2.COLOR_BGR2GRAY)
            
        # 2. Calculate Optical Flow (Farneback)
        # Parameters inspired by typical super-resolution/denoising usage
        flow = cv2.calcOpticalFlowFarneback(ref_gray, rough_gray, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, 
                                            flags=0)
                                            
        # 3. Apply Flow (Remap)
        h, w = ref_gray.shape
        
        # Create meshgrid
        flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
        y_grid, x_grid = np.indices((h, w))
        
        # Flow says "where pixel moved TO", so to pull FROM, we add flow? 
        # Actually standard Farneback: prev, next -> flow. 
        # If we want to warp 'next' (rough_aligned) to look like 'prev' (ref), 
        # flow(y,x) = (dx, dy). 
        # pixel at (y,x) in Ref comes from (y+dy, x+dx) in Next.
        
        map_x = np.float32(x_grid + flow[..., 0])
        map_y = np.float32(y_grid + flow[..., 1])
        
        aligned = cv2.remap(rough_aligned, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return aligned

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
