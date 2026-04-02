import cv2
import numpy as np

class WaveletEnhancer:
    @staticmethod
    def apply_wavelets(image, layers=None, denoise_strength=0):
        """
        Applies a simplified wavelet-style sharpening using Gaussian Blur differences.
        layers: list of (sigma, weight) tuples.
        """
        if layers is None:
            # Dyadic scales (Power of 2) are standard for wavelets (1, 2, 4, 8, 16)
            # This isolates fine vs coarse detail much better than linear steps
            layers = [(1.0, 0.0), (2.0, 0.0), (4.0, 0.0), (8.0, 0.0), (16.0, 0.0), (32.0, 0.0)]
            
        enhanced = image.astype(np.float32)
        base = enhanced.copy()
        
        # Pre-process Denoise (Optional but recommended before sharpening)
        if denoise_strength > 0:
            if len(image.shape) == 3:
                base_uint8 = base.astype(np.uint8)
                denoised = cv2.fastNlMeansDenoisingColored(base_uint8, None, denoise_strength, denoise_strength, 7, 21)
                base = denoised.astype(np.float32)
            else:
                base_uint8 = base.astype(np.uint8)
                denoised = cv2.fastNlMeansDenoising(base_uint8, None, denoise_strength, 7, 21)
                base = denoised.astype(np.float32)
        
        # This is a very simplified unsharp mask approach per layer
        # Real wavelets decompose signal, but this is a common approximation in simple tools
        # For each layer, we subtract a blur from original and add back with weight
        
        # A better approach for 'Wavelets' in planetary imaging is Difference of Gaussians
        current = base.copy()
        
        accumulated_details = np.zeros_like(base)
        
        for sigma, weight in layers:
            if weight == 0:
                continue
            
            blurred = cv2.GaussianBlur(base, (0, 0), sigma)
            # Detail at this scale
            high_freq = base - blurred
            
            accumulated_details += high_freq * weight
            
            # For next layer, we could process the blurred version or keep referencing base
            # Registax usually works on dyadic scales or linear steps.
            # Here we just add unsharp mask contributions
            
        result = base + accumulated_details
        return np.clip(result, 0, 255).astype(np.uint8)

class ColorCorrector:
    @staticmethod
    def auto_balance(image):
        """
        Simple auto white balance (Gray World assumption).
        """
        if len(image.shape) < 3:
            return image
            
        result = image.copy()
        avg_b = np.mean(result[:,:,0])
        avg_g = np.mean(result[:,:,1])
        avg_r = np.mean(result[:,:,2])
        
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        if avg_b == 0: avg_b = 1
        if avg_g == 0: avg_g = 1
        if avg_r == 0: avg_r = 1
        
        result[:,:,0] = np.clip(result[:,:,0] * (avg_gray / avg_b), 0, 255)
        result[:,:,1] = np.clip(result[:,:,1] * (avg_gray / avg_g), 0, 255)
        result[:,:,2] = np.clip(result[:,:,2] * (avg_gray / avg_r), 0, 255)
        
        return result

    @staticmethod
    def align_channels(image):
        """
        Aligns RGB channels to correct atmospheric dispersion using warp/shift.
        """
        if len(image.shape) < 3:
            return image
            
        b, g, r = cv2.split(image)
        
        # Align B and R to G
        # Using phase correlation again
        # This is expensive, might want to do it on a smaller crop
        
        from skimage.registration import phase_cross_correlation
        
        shift_b, _, _ = phase_cross_correlation(g, b, upsample_factor=10)
        shift_r, _, _ = phase_cross_correlation(g, r, upsample_factor=10)
        
        rows, cols = g.shape
        M_b = np.float32([[1, 0, shift_b[1]], [0, 1, shift_b[0]]])
        M_r = np.float32([[1, 0, shift_r[1]], [0, 1, shift_r[0]]])
        
        b_aligned = cv2.warpAffine(b, M_b, (cols, rows))
        r_aligned = cv2.warpAffine(r, M_r, (cols, rows))
        
        return cv2.merge([b_aligned, g, r_aligned])

class AutoEnhancer:
    @staticmethod
    def create_limb_mask(image):
        """
        Creates a mask of the planet limb (edge) to protect it from sharpening artifacts.
        Returns a float mask where 1.0 = protect (keep original), 0.0 = sharpen.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 1. Threshold to find planet
        # Otsu's binarization usually works well for high contrast space objects
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # 2. Erode to pull away from the edge
        kernel = np.ones((5,5), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=2)
        
        # 3. Create the "ringing zone" mask
        # We want to sharpen the center but not the very edge
        # The eroded mask covers the center.
        # Everything OUTSIDE the eroded mask (the edge + space) should be protected?
        # Actually, we want to mix:
        # Center -> Sharpened
        # Limb/Space -> Original
        
        # So mask: 0 = Center (Sharpen), 1 = Edge/Space (Protect)
        mask = np.zeros_like(gray, dtype=np.float32)
        
        # The eroded part is the safe central zone. Invert it to get the danger zone.
        safe_zone = eroded
        danger_zone = 255 - safe_zone
        
        mask = danger_zone.astype(np.float32) / 255.0
        
        # 4. Blur the mask for smooth transition
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Expand dimensions for RGB multiplication
        if len(image.shape) == 3:
            mask = cv2.merge([mask, mask, mask])
            
        return mask

    @staticmethod
    def optimize(image):
        """
        Smart automatic processing that:
        1. Applies color correction
        2. Tests multiple sharpening strengths with limb masking
        3. Selects the best result based on sharpness vs noise/clipping trade-off
        """
        # 1. First, correct the color base
        corrected = ColorCorrector.align_channels(image)
        corrected = ColorCorrector.auto_balance(corrected)
        
        # Generate protection mask
        # 1.0 = Protect (keep original), 0.0 = Sharpen
        limb_mask = AutoEnhancer.create_limb_mask(corrected)
        
        # 2. Define search space for sharpening
        # Base layers emphasize mid-detail (layers 2-4) common in planets
        base_layers = [
            (1.0, 0.0), 
            (2.0, 1.0), 
            (3.0, 1.0), 
            (4.0, 0.5), 
            (5.0, 0.0), 
            (6.0, 0.0)
        ]
        
        strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        
        best_score = -1.0
        best_image = corrected
        
        # Calculate base stats
        base_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
        
        # Calculate 'clipping' roughly
        # We want to avoid pushing too many pixels to 0 or 255 if they weren't there
        
        for s in strengths:
            # Scale weights
            current_layers = [(sigma, w * s) for sigma, w in base_layers]
            
            # Apply Wavelets
            sharpened_full = WaveletEnhancer.apply_wavelets(corrected, current_layers)
            
            # Apply Limb Masking to prevent ringing
            # Result = Original * Mask + Sharpened * (1 - Mask)
            attempt_float = corrected.astype(np.float32) * limb_mask + sharpened_full.astype(np.float32) * (1.0 - limb_mask)
            attempt = np.clip(attempt_float, 0, 255).astype(np.uint8)
            
            # Analyze
            if len(attempt.shape) == 3:
                gray = cv2.cvtColor(attempt, cv2.COLOR_BGR2GRAY)
            else:
                gray = attempt
                
            # Metric 1: Sharpness (Variance of Laplacian)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Metric 2: Clipping Penalty
            # Calculate % of pixels at limits
            lower_clip = np.sum(gray <= 0)
            upper_clip = np.sum(gray >= 255)
            total_pixels = gray.size
            clip_ratio = (lower_clip + upper_clip) / total_pixels
            
            # Simple heuristic score:
            # We want high sharpness, but if clipping > 1% (0.01), penalize heavily
            if clip_ratio > 0.02: # Tolerant up to 2%
                penalty = 0.5 # Halve the score effectively
            else:
                penalty = 1.0
                
            score = sharpness * penalty
            
            if score > best_score:
                best_score = score
                best_image = attempt
                
        return best_image
