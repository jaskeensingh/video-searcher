# Generative AI and Computer Vision Video Searcher

## Overview
A sophisticated multi-modal AI system that enhances video searchability by automatically generating comprehensive metadata through the analysis of video content, audio, and text. The system combines state-of-the-art techniques in Computer Vision, Natural Language Processing, and Audio Analysis to make videos searchable even with minimal initial metadata.

## Features
- **Video Frame Analysis**: Extracts and analyzes key frames to identify objects, scenes, and activities
- **Optical Character Recognition (OCR)**: Detects and extracts text appearing in videos
- **Audio Transcription**: Converts speech to text for enhanced searchability
- **Object Detection**: Identifies and tags objects and people in video frames
- **Scene Understanding**: Classifies different scenes and settings
- **Metadata Generation**: Creates rich, searchable metadata from all analyzed components
- **Multi-Parameter Search**: Enables searching across all generated metadata types

## Technical Architecture
The system is built using Python and leverages several key technologies:

- **Computer Vision**: OpenCV, YOLOv8
- **OCR**: Tesseract OCR
- **Speech Recognition**: Whisper
- **Natural Language Processing**: Sentence-Transformers
- **Deep Learning Framework**: PyTorch
- **Video Processing**: FFmpeg
- **Database**: PostgreSQL with vector extensions

## Installation

```bash
# Clone the repository
git clone https://github.com/jaskeensingh/video-searcher.git
cd video-searcher

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies
# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install ffmpeg tesseract-ocr
```

## Usage

```python
from video_searcher import VideoAnalyzer

# Initialize the analyzer
analyzer = VideoAnalyzer(model_path='models/')

# Process a video file
metadata = analyzer.process_video('path/to/video.mp4')

# Search through processed videos
results = analyzer.search('person walking in park')
```

## Project Structure
```
video-searcher/
├── src/
│   ├── video_processor/
│   │   ├── frame_extractor.py
│   │   ├── object_detector.py
│   │   └── scene_analyzer.py
│   ├── text_processor/
│   │   ├── ocr_engine.py
│   │   └── text_analyzer.py
│   ├── audio_processor/
│   │   ├── speech_recognizer.py
│   │   └── audio_analyzer.py
│   └── search_engine/
│       ├── indexer.py
│       └── searcher.py
├── models/
├── tests/
├── docs/
└── examples/
```
