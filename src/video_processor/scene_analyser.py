from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class SceneAnalyzer:
    def __init__(self):
        """Initialize scene analyzer with pre-trained model"""
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def analyze_scene(self, frame):
        """
        Analyze scene content and return scene description
        """
        # Convert frame to format expected by model
        image_features = self.model.encode_image(frame)
        
        # Get scene embeddings
        scene_embedding = image_features / np.linalg.norm(image_features)
        
        return {
            'embedding': scene_embedding,
            'features': image_features
        }