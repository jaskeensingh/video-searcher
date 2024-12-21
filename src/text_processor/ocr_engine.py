import pytesseract
import cv2
import numpy as np

class OCREngine:
    def __init__(self):
        """
        Initialize OCR engine with Tesseract
        """
        self.config = r'--oem 3 --psm 6'
        
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        thresh = cv2.threshold(gray, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opening
        
    def extract_text(self, image):
        """
        Extract text from image using OCR
        """
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        # Extract text
        text = pytesseract.image_to_string(processed_img, config=self.config)
        
        # Get bounding boxes for text regions
        boxes = pytesseract.image_to_data(processed_img, config=self.config, 
                                        output_type=pytesseract.Output.DICT)
        
        results = []
        n_boxes = len(boxes['text'])
        for i in range(n_boxes):
            if int(boxes['conf'][i]) > 60:  # Filter low confidence
                results.append({
                    'text': boxes['text'][i],
                    'confidence': float(boxes['conf'][i]),
                    'bbox': [boxes['left'][i], boxes['top'][i], 
                            boxes['width'][i], boxes['height'][i]]
                })
                
        return results