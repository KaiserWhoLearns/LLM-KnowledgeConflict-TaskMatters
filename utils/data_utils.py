
import os
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import Repository
from tqdm import tqdm
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

def combine_into_pairs():
    """
    Convert conflictqa into the format of WikiContradict
    """
    new_data = []
    print(os.environ["HF_HOME"])
    conflictqa = load_dataset("KaiserWhoLearns/conflictqa-u", split="train")
    # Gather the set of all question
    all_questions = conflictqa["search_query"]

    # For each question, gather a yes and a no evidence
    for q in all_questions:
        yes_evidence = conflictqa.filter(lambda example: example["search_query"] == q and example["stance"] == "yes")
        no_evidence = conflictqa.filter(lambda example: example["search_query"] == q and example["stance"] == "no")
        for yes_instance, no_instance in zip(yes_evidence, no_evidence):
            new_data.append({
                "question": q,
                "context1": yes_instance["text_window"],
                "context2": no_instance["text_window"],
                "answer1": "yes",
                "answer2": "no",
            })
    return Dataset.from_list(new_data)

def combine_datasets():
    """
    dataset_name in {conflictqa, druid, wikicontradict}
    Organize the data into Question - Plausible Evidence - PE-answer
    such that we can generate inplausible evidence
    """
    # Convert conflictQA into WikiContradict format
    conflictqa = combine_into_pairs()
    wikicontradict = load_dataset("ibm-research/Wikipedia_contradict_benchmark", split="train")
    # Remove irrelevant columns
    merged_data = concatenate_datasets([wikicontradict, conflictqa])
    # Define the columns to keep
    columns_to_keep = ["question", "context1", "context2", "answer1", "answer2"]

    # Remove other columns
    merged_data = merged_data.remove_columns([col for col in merged_data.column_names if col not in columns_to_keep])
    merged_data.save_to_disk(os.path.join(os.environ["data_dir"], "intermediate_processing", "merged_raw"))


if __name__ == "__main__":
    # Before running, download the conflictQA data using wget
    # wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yPxzZsKzT96I6K_WLKDDxYTUZ9gQHZoU' -O conflictqa.pkl
    get_constant()

    ##### Convert ConflictQA to HF Dataset
    # Specify the path to the pandas pickle file
    pkl_file_path = os.path.join(os.environ["data_dir"], "conflictqa.pkl")
    # Optionally specify where to save the Hugging Face dataset
    save_to_path = os.path.join(os.environ["data_dir"], "conflictqa")
    
    convert_to_huggingface_dataset(pkl_file_path, save_to=save_to_path)


    ##### Merge ConflictQA with wikiContradict
    combine_datasets()
    