# Sample 50 instances for human annotation

from datasets import load_dataset
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # Path to your .jsonl file
    jsonl_path = os.path.join(os.environ["data_dir"], "final_data_filtered", "llama3.2-3B-Instruct_full_v2.jsonl")
    # "path/to/your/data.jsonl"

    # Load the .jsonl file as a Hugging Face dataset
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    # Annotation: LPC Validity and HPCE Explanation

    # Columns to keep
    columns_to_keep = ["question", "NC_context", "NC_answer","HPC_context", "HPC_answer", "HPCE_context", "HPCE_answer", "LPC_context", "LPC_answer"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)

    # Sample 50 instances
    sampled_dataset = dataset.shuffle(seed=42).select(range(50))

    # Convert to a pandas DataFrame
    df = pd.DataFrame(sampled_dataset)

    # Ensure output directory exists
    output_dir = os.path.join(os.environ["base_dir"], "output", "annotation")
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(output_dir, "annotation.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved 50 samples to {output_path}")
