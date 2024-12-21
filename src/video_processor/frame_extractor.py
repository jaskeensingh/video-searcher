import cv2
import numpy as np
from pathlib import Path
import logging

class FrameExtractor:
    def __init__(self, sample_rate=1):
        """
        Initialize frame extractor with sampling rate (frames per second)
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    def extract_frames(self, video_path):
        """
        Extract frames from video at specified sample rate
        Returns list of frames as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.sample_rate)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                frame_count += 1

            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            raise