# Sample 50 instances for human annotation

from datasets import load_dataset
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()

def sample_for_MBE_agreement():
    """
    For PK, PCK, CK, RAG, sample 10 examples each for agreement analysis
    """
    # for task in ["CK", "PK", "PCK", "RAG"]:
    for task in ["PK"]:
        # Load the metric json
        jsonl_path = os.path.join(os.environ["base_dir"], "output", "metrics", f"llama3.2-3B-Instruct_{task}_full_v2.jsonl")
        dataset = load_dataset("json", data_files=jsonl_path, split="train")
        sampled_dataset = dataset.shuffle(seed=42).select(range(10))

        # Convert to a pandas DataFrame
        df = pd.DataFrame(sampled_dataset)
        df = pd.concat([df.drop("metrics", axis=1), df["metrics"].apply(pd.Series)], axis=1)

        # Ensure output directory exists
        output_dir = os.path.join(os.environ["base_dir"], "output", "annotation", "eval_agreement")
        os.makedirs(output_dir, exist_ok=True)

        # Write data
        output_path = os.path.join(output_dir, f"{task}.csv")
        df.to_csv(output_path, index=False)

def sample_for_MBE_agreement_by_evidence_type(evidence_type):
    """
    For PK, PCK, CK, RAG, sample 10 examples each for agreement analysis
    """
    # for task in ["CK", "PK", "PCK", "RAG"]:
    for task in ["RAG"]:
        # Load the metric json
        jsonl_path = os.path.join(os.environ["base_dir"], "output", "metrics", f"llama3.2-3B-Instruct_{task}_full_v2.jsonl")
        dataset = load_dataset("json", data_files=jsonl_path, split="train")
        dataset = dataset.filter(lambda x: x["context_type"] == evidence_type)
        sampled_dataset = dataset.shuffle(seed=42).select(range(10))

        # Convert to a pandas DataFrame
        df = pd.DataFrame(sampled_dataset)
        df = pd.concat([df.drop("metrics", axis=1), df["metrics"].apply(pd.Series)], axis=1)

        # Ensure output directory exists
        output_dir = os.path.join(os.environ["base_dir"], "output", "annotation", "eval_agreement")
        os.makedirs(output_dir, exist_ok=True)

        # Write data
        output_path = os.path.join(output_dir, f"{task}_{evidence_type}.csv")
        df.to_csv(output_path, index=False)

def sample_for_evidence_annotation():
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

        

if __name__ == "__main__":
    # sample_for_MBE_agreement()
    # sample_for_evidence_annotation()
    sample_for_MBE_agreement_by_evidence_type(evidence_type="NC")
    sample_for_MBE_agreement_by_evidence_type(evidence_type="LPC")
    sample_for_MBE_agreement_by_evidence_type(evidence_type="HPCE")
    sample_for_MBE_agreement_by_evidence_type(evidence_type="HPC")