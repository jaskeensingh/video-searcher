from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
from typing import List, Dict

class Searcher:
    def __init__(self, db_config: Dict = None):
        """Initialize searcher"""
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_config = db_config or {
            'dbname': 'video_search',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost'
        }
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for videos matching the query
        Returns list of matching videos with relevance scores
        """
        # Encode search query
        query_embedding = self.text_model.encode(query)
        
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor() as cur:
                # Search using vector similarity
                cur.execute("""
                    SELECT v.path, v.metadata, e.embedding_type,
                           e.metadata, (e.embedding <-> %s) as distance
                    FROM videos v
                    JOIN embeddings e ON v.id = e.video_id
                    ORDER BY distance ASC
                    LIMIT %s
                """, (query_embedding.tolist(), limit))
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        'video_path': row[0],
                        'video_metadata': row[1],
                        'match_type': row[2],
                        'match_metadata': row[3],
                        'relevance_score': 1 - row[4]  # Convert distance to similarity
                    })
                
                return results