import cv2
import numpy as np

class WaveletEnhancer:
    @staticmethod
    def apply_wavelets(image, layers=None):
        """
        Applies a simplified wavelet-style sharpening using Gaussian Blur differences.
        layers: list of (sigma, weight) tuples.
        """
        if layers is None:
            # Default layers roughly mimicking Registax 6 layers logic locally
            layers = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0), (6.0, 0.0)]
            
        enhanced = image.astype(np.float32)
        base = enhanced.copy()
        
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
