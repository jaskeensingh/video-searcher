import pytest
import numpy as np
from src.text_processor.ocr_engine import OCREngine

def test_ocr_engine():
    ocr = OCREngine()
    
    # Create test image with text
    image = np.zeros((100, 300, 3), dtype=np.uint8)
    # Add text to image...
    
    results = ocr.extract_text(image)
    assert isinstance(results, list)