import os
import wget
import torch
from sentence_transformers import SentenceTransformer
import whisper
from pathlib import Path

def create_directories():
    """Create model directories if they don't exist"""
    directories = [
        'models/object_detection',
        'models/scene_understanding',
        'models/speech_recognition',
        'models/text_embedding'
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_yolo():
    """Download YOLOv8 model"""
    print("Downloading YOLOv8 model...")
    yolo_path = 'models/object_detection/yolov8n.pt'
    if not os.path.exists(yolo_path):
        wget.download(
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            yolo_path
        )
    print("\nYOLOv8 model downloaded successfully!")

def download_clip():
    """Download and save CLIP model"""
    print("Downloading CLIP model...")
    clip_path = 'models/scene_understanding/clip-ViT-B-32'
    if not os.path.exists(clip_path):
        model = SentenceTransformer('clip-ViT-B-32')
        model.save(clip_path)
    print("CLIP model downloaded successfully!")

def download_whisper():
    """Download Whisper model"""
    print("Downloading Whisper model...")
    whisper_path = 'models/speech_recognition/whisper-base.pt'
    if not os.path.exists(whisper_path):
        model = whisper.load_model('base')
        # Whisper handles downloading automatically
    print("Whisper model downloaded successfully!")

def download_text_embedding():
    """Download text embedding model"""
    print("Downloading text embedding model...")
    embedding_path = 'models/text_embedding/all-MiniLM-L6-v2'
    if not os.path.exists(embedding_path):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.save(embedding_path)
    print("Text embedding model downloaded successfully!")

def main():
    print("Starting model downloads...")
    create_directories()
    download_yolo()
    download_clip()
    download_whisper()
    download_text_embedding()
    print("\nAll models downloaded successfully!")

if __name__ == '__main__':
    main()