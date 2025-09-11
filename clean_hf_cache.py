#!/usr/bin/env python3
"""Clean corrupted HuggingFace cache files to fix 416 errors"""

import os
import sys
import shutil
from huggingface_hub import constants

def clean_model_cache(model_name):
    """Remove corrupted cache for a specific model"""
    cache_dir = constants.HF_HUB_CACHE
    
    # Convert model name to cache directory format
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    
    model_path = os.path.join(cache_dir, model_dir_name)
    lock_path = os.path.join(cache_dir, ".locks", model_dir_name)
    
    cleaned = False
    
    if os.path.exists(model_path):
        print(f"Removing cache for {model_name} at: {model_path}")
        shutil.rmtree(model_path)
        print("✓ Model cache cleaned")
        cleaned = True
    
    if os.path.exists(lock_path):
        print(f"Removing lock files at: {lock_path}")
        shutil.rmtree(lock_path)
        print("✓ Lock files cleaned")
        cleaned = True
    
    if not cleaned:
        print(f"No cache found for model: {model_name}")
    
    return cleaned

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_hf_cache.py <model_name>")
        print("Example: python clean_hf_cache.py mistralai/Mistral-7B-Instruct-v0.3")
        print("\nOr use 'all' to see all cached models:")
        print("python clean_hf_cache.py all")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    if model_name == "all":
        cache_dir = constants.HF_HUB_CACHE
        print(f"Cached models in {cache_dir}:")
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                if item.startswith("models--"):
                    model = item.replace("models--", "").replace("--", "/")
                    size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(os.path.join(cache_dir, item))
                              for filename in filenames) / (1024**3)  # GB
                    print(f"  - {model} ({size:.2f} GB)")
    else:
        if clean_model_cache(model_name):
            print(f"\n✓ Successfully cleaned cache for {model_name}")
            print("The model will be re-downloaded on next use.")
        else:
            print(f"\n✗ No action taken for {model_name}")

if __name__ == "__main__":
    main()