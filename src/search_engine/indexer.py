from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
import json
from typing import List, Dict

class Indexer:
    def __init__(self, db_config: Dict = None):
        """Initialize search indexer"""
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_config = db_config or {
            'dbname': 'video_search',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost'
        }
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Create tables if they don't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS videos (
                        id SERIAL PRIMARY KEY,
                        path TEXT NOT NULL,
                        metadata JSONB
                    );
                    
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        video_id INTEGER REFERENCES videos(id),
                        embedding_type TEXT,
                        embedding vector(384),
                        metadata JSONB
                    );
                """)
    
    def index_video(self, metadata: Dict):
        """Index video metadata in database"""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Insert video metadata
                cur.execute(
                    "INSERT INTO videos (path, metadata) VALUES (%s, %s) RETURNING id",
                    (metadata['video_path'], json.dumps(metadata))
                )
                video_id = cur.fetchone()[0]
                
                # Create embeddings for searchable content
                self._index_objects(cur, video_id, metadata['objects'])
                self._index_text(cur, video_id, metadata['text'])
                self._index_speech(cur, video_id, metadata['speech'])
    
    def _index_objects(self, cur, video_id: int, objects: List[Dict]):
        """Index object detections"""
        for obj in objects:
            text = f"Frame {obj['frame']}: {obj['class']}"
            embedding = self.text_model.encode(text)
            cur.execute(
                """INSERT INTO embeddings 
                   (video_id, embedding_type, embedding, metadata)
                   VALUES (%s, %s, %s, %s)""",
                (video_id, 'object', embedding.tolist(), json.dumps(obj))
            )
    
    def _index_text(self, cur, video_id: int, texts: List[Dict]):
        """Index OCR text"""
        for text in texts:
            embedding = self.text_model.encode(text['text'])
            cur.execute(
                """INSERT INTO embeddings
                   (video_id, embedding_type, embedding, metadata)
                   VALUES (%s, %s, %s, %s)""",
                (video_id, 'text', embedding.tolist(), json.dumps(text))
            )
    
    def _index_speech(self, cur, video_id: int, speeches: List[Dict]):
        """Index speech transcriptions"""
        for speech in speeches:
            embedding = self.text_model.encode(speech['text'])
            cur.execute(
                """INSERT INTO embeddings
                   (video_id, embedding_type, embedding, metadata)
                   VALUES (%s, %s, %s, %s)""",
                (video_id, 'speech', embedding.tolist(), json.dumps(speech))
            )