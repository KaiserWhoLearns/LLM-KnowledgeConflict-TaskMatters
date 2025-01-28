
import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import Repository
import pdb

from constant import get_constant


def convert_to_huggingface_dataset(pkl_file_path, save_to=None):
    # Convert ConflictQA to HF dataset
    df = pd.read_pickle(pkl_file_path)
    hf_dataset = Dataset.from_pandas(df)
    if save_to:
        try:
            hf_dataset.save_to_disk(save_to)
            print(f"Hugging Face Dataset saved to {save_to}.")
        except Exception as e:
            print(f"Error saving the dataset: {e}")
        hf_dataset.push_to_hub("KaiserWhoLearns/conflictqa-u")

if __name__ == "__main__":
    # Before running, download the conflictQA data using wget
    # wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yPxzZsKzT96I6K_WLKDDxYTUZ9gQHZoU' -O conflictqa.pkl
    get_constant()
    # Specify the path to the pandas pickle file
    pkl_file_path = os.path.join(os.environ["data_dir"], "conflictqa.pkl")
    # Optionally specify where to save the Hugging Face dataset
    save_to_path = os.path.join(os.environ["data_dir"], "conflictqa")
    
    convert_to_huggingface_dataset(pkl_file_path, save_to=save_to_path)
    