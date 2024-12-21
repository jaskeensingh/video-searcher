import whisper
import torch
import numpy as np

class SpeechRecognizer:
    def __init__(self, model_name='base'):
        """Initialize Whisper speech recognition model"""
        self.model = whisper.load_model(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def transcribe(self, video_path):
        """
        Transcribe speech from video audio
        Returns list of transcriptions with timestamps
        """
        result = self.model.transcribe(video_path)
        
        transcriptions = []
        for segment in result['segments']:
            transcriptions.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            })
            
        return transcriptions