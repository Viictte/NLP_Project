#!/usr/bin/env python3
"""
Helper script to download the embedding model for offline use.

This script downloads the BAAI/bge-m3 embedding model to a local directory,
allowing you to use the RAG system without needing to connect to HuggingFace
on every run.

Usage:
    python download_embedding_model.py [output_directory]

If no output directory is specified, the model will be downloaded to:
    ./models/bge-m3

After downloading, set EMBEDDING_MODEL_PATH in your .env file:
    EMBEDDING_MODEL_PATH=/path/to/models/bge-m3
"""

import sys
import os
from pathlib import Path

def download_model(output_dir: str = "./models/bge-m3"):
    """Download the embedding model to a local directory"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading BAAI/bge-m3 embedding model to: {output_path}")
    print("This will download approximately 1-2GB of data...")
    print()
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download the model (this will cache it in HuggingFace cache first)
        print("Step 1: Downloading model from HuggingFace...")
        model = SentenceTransformer("BAAI/bge-m3")
        
        # Save to the specified directory
        print(f"Step 2: Saving model to {output_path}...")
        model.save(str(output_path))
        
        print()
        print("=" * 60)
        print("✓ Model downloaded successfully!")
        print("=" * 60)
        print()
        print("To use this model in offline mode:")
        print("1. Add this line to your .env file:")
        print(f"   EMBEDDING_MODEL_PATH={output_path}")
        print()
        print("2. (Optional) Enable full offline mode:")
        print("   HF_HUB_OFFLINE=1")
        print()
        print("3. Run your RAG system as normal - it will use the local model")
        print()
        
        return True
        
    except ImportError:
        print("Error: sentence-transformers is not installed")
        print("Please install it with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Try using a VPN if HuggingFace is blocked in your region")
        print("3. Try using a mirror: export HF_ENDPOINT=https://hf-mirror.com")
        print("4. Increase timeout: export HF_HUB_TIMEOUT=120")
        return False

def main():
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "./models/bge-m3"
    
    success = download_model(output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
