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
        cx = max(half_w, min(frame_w - (w - half_w), cx))
        cy = max(half_h, min(frame_h - (h - half_h), cy))
        
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
        Estimates image sharpness using the mean magnitude of Sobel gradients.
        This is more robust to noise than Laplacian variance and better represents feature contrast.
        """
        if frame is None:
            return 0.0
            
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Sobel Gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = cv2.magnitude(gx, gy)
        
        return np.mean(magnitude)

    @staticmethod
    def analyze_quality_graph(qualities):
        """
        Analyzes the sorted quality scores to find the optimal stacking percentage.
        Uses a 'Knee Detection' algorithm (Kneedle or simple acceleration analysis).
        Returns the recommended percentage (1-100).
        """
        if qualities is None or len(qualities) < 10:
            return 50 # Fallback
            
        # Qualities are assumed unsorted here, so sort them descending
        sorted_q = np.sort(qualities)[::-1]
        
        # Normalize
        y = (sorted_q - np.min(sorted_q)) / (np.max(sorted_q) - np.min(sorted_q) + 1e-6)
        x = np.linspace(0, 1, len(y))
        
        # Simple Knee Finder: Max distance from the line connecting first and last point
        # Line equation: Ax + By + C = 0
        # P1=(0, y[0]), P2=(1, y[-1])
        # Since y is decreasing, P1=(0, 1), P2=(1, 0) roughly if fully normalized, 
        # but here y[0]=1, y[-1]=0.
        
        # Vector from start to end
        vec_s_e = np.array([1.0, y[-1] - y[0]])
        vec_s_e_norm = np.linalg.norm(vec_s_e)
        
        distances = []
        for i, val in enumerate(y):
            # Point P
            px, py = x[i], val
            
            # Distance from P to line defined by P_start(0, y[0]) and P_end(1, y[-1])
            # d = |(x_e - x_s)(y_s - y_p) - (x_s - x_p)(y_e - y_s)| / length
            # Simplifies since x_s=0
            
            num = np.abs(1.0 * (y[0] - py) - (-x[i]) * (y[-1] - y[0]))
            d = num / vec_s_e_norm
            distances.append(d)
            
        # The knee is the point of maximum distance
        knee_idx = np.argmax(distances)
        
        # Convert index to percentage
        # Safeties: at least 5%, at most 80% usually
        pct = (knee_idx / len(qualities)) * 100.0
        
        # Heuristic adjustments
        # If knee is too early (bad data), take at least 10%
        pct = max(10, min(90, pct))
        
        return int(pct)

    @staticmethod
    def recognize_object(frame):
        """
        Attempts to recognize the celestial object in the given cropped/stacked frame 
        using basic heuristics like aspect ratio and color channels.
        """
        if frame is None:
            return "Unknown"

        # Check if the image is colored
        is_color = len(frame.shape) == 3
        
        # Convert to grayscale for shape analysis
        if is_color:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Find the object using a threshold
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        aspect_ratio = 1.0
        area = 0
        if contours:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            _, _, w, h = cv2.boundingRect(c)
            if h > 0:
                aspect_ratio = w / h
                
        # If the object takes up a huge portion of the frame and is relatively flat
        # it might be the Moon surface
        frame_area = frame.shape[0] * frame.shape[1]
        if area > frame_area * 0.8:
            return "Moon (Surface)"

        # Saturn: Wide aspect ratio due to rings
        if aspect_ratio > 1.3 or aspect_ratio < 0.7:
            return "Saturn"

        if is_color:
            # Color based heuristics
            mean_color = cv2.mean(frame, mask=thresh)[:3] # B, G, R
            b, g, r = mean_color

            # Mars is notably red
            if r > g * 1.3 and r > b * 1.3:
                return "Mars"
            
            # Jupiter has strong bands but generally a warm brownish/yellowish hue
            # This is a generic heuristic:
            if r > b and g > b:
                return "Jupiter / Venus"
        else:
            # Grayscale fallback
            if aspect_ratio >= 0.85 and aspect_ratio <= 1.15:
                return "Planet (Jupiter/Mars/Venus)"

        return "Unknown Celestial Body"
