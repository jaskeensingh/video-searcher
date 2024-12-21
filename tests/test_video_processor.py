import pytest
import numpy as np
from src.video_processor.frame_extractor import FrameExtractor
from src.video_processor.object_detector import ObjectDetector

def test_frame_extractor():
    extractor = FrameExtractor(sample_rate=1)
    
    # Test with sample video
    frames = extractor.extract_frames('tests/data/sample.mp4')
    assert len(frames) > 0
    assert isinstance(frames[0], np.ndarray)
    
def test_object_detector():
    detector = ObjectDetector()
    
    # Create dummy frame
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    
    # Test detection
    detections = detector.detect_objects(frame)
    assert isinstance(detections, list)