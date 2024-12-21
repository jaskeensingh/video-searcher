from ultralytics import YOLO
import torch
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLO object detector
        """
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def detect_objects(self, frame):
        """
        Detect objects in a single frame
        Returns list of detected objects with bounding boxes and confidence scores
        """
        results = self.model(frame, device=self.device)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = box.cls[0]
                class_name = self.model.names[int(class_id)]
                
                detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
                
        return detections