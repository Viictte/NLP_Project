"""Embedding service using bge-m3"""

from typing import List, Union
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from rag_system.core.config import get_config

# Configure HuggingFace Hub for better reliability in regions with poor connectivity
try:
    from huggingface_hub import constants as hf_constants
    
    # Increase HTTP timeout for model downloads (default is 10s, increase to 60s)
    # This helps users in regions with slow/unreliable connections to HuggingFace
    timeout = int(os.getenv('HF_HUB_TIMEOUT', '60'))
    hf_constants.HF_HUB_HTTP_TIMEOUT = timeout
    
    # Support custom HuggingFace endpoint for mirrors/proxies
    # Users can set HF_ENDPOINT to use a mirror (e.g., https://hf-mirror.com)
    hf_endpoint = os.getenv('HF_ENDPOINT')
    if hf_endpoint:
        hf_constants.HF_ENDPOINT = hf_endpoint
        print(f"Using custom HuggingFace endpoint: {hf_endpoint}")
except Exception as e:
    print(f"Warning: Could not configure HuggingFace Hub settings: {e}")

class EmbeddingService:
    def __init__(self):
        config = get_config()
        model_name = config.get('embeddings.model', 'BAAI/bge-m3')
        self.batch_size = config.get('embeddings.batch_size', 32)
        
        # Support for local model path (offline mode)
        local_model_path = os.getenv('EMBEDDING_MODEL_PATH')
        
        try:
            if local_model_path and os.path.exists(local_model_path):
                # Load from local directory without accessing HuggingFace
                print(f"Loading embedding model from local path: {local_model_path}")
                print("(Offline mode - no HuggingFace connection required)")
                # When given a local path, SentenceTransformer loads from disk without network access
                self.model = SentenceTransformer(local_model_path)
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"Embedding model loaded successfully (dimension: {self.dimension})")
            else:
                # Normal online mode - download from HuggingFace if needed
                print(f"Loading embedding model: {model_name}")
                print("This may take a few minutes on first run (downloading ~1-2GB)...")
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"Embedding model loaded successfully (dimension: {self.dimension})")
        except Exception as e:
            # Provide context-aware error message
            if local_model_path:
                error_msg = (
                    f"Failed to load embedding model from local path '{local_model_path}': {str(e)}\n\n"
                    "Troubleshooting:\n"
                    "1. Verify the model directory exists and contains valid model files\n"
                    "2. Try re-downloading the model: python download_embedding_model.py\n"
                    "3. Check that EMBEDDING_MODEL_PATH points to the correct directory\n"
                )
            else:
                error_msg = (
                    f"Failed to load embedding model '{model_name}': {str(e)}\n\n"
                    "If you're experiencing connection timeouts to huggingface.co:\n"
                    "1. Use offline mode with a pre-downloaded model:\n"
                    "   - Download the model once on a machine with good connectivity\n"
                    "   - Set EMBEDDING_MODEL_PATH=/path/to/local/model in .env\n"
                    "   - Optionally set HF_HUB_OFFLINE=1 for full offline mode\n"
                    "2. Try increasing the timeout: export HF_HUB_TIMEOUT=120\n"
                    "3. Use a VPN or proxy to access huggingface.co\n"
                    "4. Use a mirror: export HF_ENDPOINT=https://hf-mirror.com\n\n"
                    "See README > Troubleshooting > Model Downloads for more details."
                )
            raise RuntimeError(error_msg) from e
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text(query)

_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service