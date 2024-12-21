import pytest
from src.audio_processor.speech_recognizer import SpeechRecognizer

def test_speech_recognizer():
    recognizer = SpeechRecognizer()
    
    # Test with sample audio
    transcriptions = recognizer.transcribe('tests/data/sample.mp3')
    assert isinstance(transcriptions, list)
    assert all('text' in t for t in transcriptions)