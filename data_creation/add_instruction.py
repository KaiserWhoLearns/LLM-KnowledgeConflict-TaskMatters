# Add instruction data
import re
import os
import sys
import re
import pdb
sys.path.append(os.getcwd())
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset
load_dotenv()

CONTEXT_TYPES = ["NC", "HPC", "HPCE", "LPC"]
def knowledge_free_tasks(raw_dataset):
    # Create knowledge free tasks data
    system_prompt = "Count the number of characters in the given context, you should only count Latin character from A to Z (both upper and lower cases). Punctuations, spaces, and utf-8 characters are not included. For example:" + \
        "Input: This is a sentence. Output: 15\n" + \
        "Input: a p p l e Output: 5\n"
    def create_kf_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_KF_input"] = system_prompt + "Input: " + example[f"{context_type}_context"] + "\nOutput: "
            example[f"{context_type}_KF_output"] = len(re.findall(r'[A-Za-z]', example[f"{context_type}_context"]))
        return example
    processed_dataset = raw_dataset.map(create_kf_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_knowledge_free.jsonl"))
    return processed_dataset


def contextual_knowledge_tasks(raw_dataset):
    # Create knowledge free tasks data
    prompt = "Answer the question solely base on the contextual data."
    pass

def parametric_knowledge_tasks(raw_dataset):
    # Create knowledge free tasks data
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    args = parser.parse_args()
    model_name = args.test_model_name
    # Load dataset
    raw_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_strictPCE.jsonl"))["train"]

    knowledge_free_tasks(raw_dataset)