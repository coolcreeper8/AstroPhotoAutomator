import cv2
import numpy as np

class VideoLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {filepath}")
            
    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
        
    def get_frame(self, index=None):
        """
        Get a specific frame or the next frame.
        If index is provided, seeks to that frame.
        """
        if index is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def load_all_frames(self, max_frames=None):
        """
        Loads all frames into memory (be careful with large videos).
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            if max_frames and len(frames) >= max_frames:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
            
        return frames

    def release(self):
        self.cap.release()
