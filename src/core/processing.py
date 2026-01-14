import cv2
import numpy as np
from scipy.ndimage import center_of_mass

class FrameAnalyzer:
    @staticmethod
    def detect_roi(frame, threshold=20):
        """
        Detects the center of the planet in the frame.
        Assumes the planet is brighter than the background.
        """
        if frame is None:
            return None
            
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Simple thresholding
        _, val = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate center of mass
        cy, cx = center_of_mass(val)
        
        if np.isnan(cx) or np.isnan(cy):
            # Fallback to image center if nothing detected
            h, w = gray.shape
            return (w // 2, h // 2)
            
        return (int(cx), int(cy))

    @staticmethod
    def crop_centered(frame, center, size):
        """
        Crops the frame around the center with given size (width, height).
        Ensures the output always has exactly the requested dimensions.
        """
        cx, cy = center
        w, h = size
        
        frame_h, frame_w = frame.shape[:2]
        
        # Adjust center to ensure crop window stays within frame bounds
        # This guarantees consistent output dimensions
        half_w = w // 2
        half_h = h // 2
        
        # Clamp center position to keep crop window fully within image
        cx = max(half_w, min(frame_w - half_w, cx))
        cy = max(half_h, min(frame_h - half_h, cy))
        
        # Calculate crop boundaries
        x1 = cx - half_w
        y1 = cy - half_h
        x2 = x1 + w
        y2 = y1 + h
        
        # Handle edge case where requested size is larger than frame
        if w > frame_w or h > frame_h:
            # Crop what we can and pad with black
            actual_w = min(w, frame_w)
            actual_h = min(h, frame_h)
            x1 = max(0, (frame_w - actual_w) // 2)
            y1 = max(0, (frame_h - actual_h) // 2)
            x2 = x1 + actual_w
            y2 = y1 + actual_h
            
            cropped = frame[y1:y2, x1:x2]
            
            # Pad to requested size
            if len(frame.shape) == 3:
                padded = np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
                pad_y = (h - actual_h) // 2
                pad_x = (w - actual_w) // 2
                padded[pad_y:pad_y+actual_h, pad_x:pad_x+actual_w] = cropped
            else:
                padded = np.zeros((h, w), dtype=frame.dtype)
                pad_y = (h - actual_h) // 2
                pad_x = (w - actual_w) // 2
                padded[pad_y:pad_y+actual_h, pad_x:pad_x+actual_w] = cropped
            
            return padded
        
        return frame[y1:y2, x1:x2]

    @staticmethod
    def estimate_quality(frame):
        """
        Estimates image sharpness using the variance of the Laplacian.
        Higher is better.
        """
        if frame is None:
            return 0.0
            
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        return cv2.Laplacian(gray, cv2.CV_64F).var()
