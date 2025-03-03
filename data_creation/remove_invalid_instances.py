import os
import sys
import re
import pdb
sys.path.append(os.getcwd())
import json
import time
import requests
import openai
import argparse
from openai import OpenAI
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from dotenv import load_dotenv

from utils.constant import get_constant
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
EDITOR_MODEL_NAME = "gpt-4o"

def is_valid(context, question, answer):
    """
    Return true if the given context-question-answer pair is valid (the context lead to the answer of the given question)
    """
    prompt = f"Question: {question}\nWith the passage below, output 'yes' if you are able to confirm that the answer to the given question is '{answer}'. Output 'no' otherwise. You should only output 'yes' or 'no'.\n{context}"

    completion = client.chat.completions.create(
            model=EDITOR_MODEL_NAME,
            messages=[
                {"role": "developer", "content": ""},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
    )
    response = completion.choices[0].message.content
    return True if "yes" in response else False

def remove_invalid_instances(dataset):
    valid_data = []
    # Loop through the dataset
    for instance in dataset:
        # Check whether LPC is LPC enough
        # pdb.set_trace()
        prompt_lpc = f"You are a experienced and wise scholar. Your job is to rate from 1-5 on whether the story given in the passage is likely to happen or not based on real-world knowledge. You should only output the scores without any justification, with 1 indicates least likely to happen, and 5 to be most likely to happen.\n {instance['LPC_context']}"
        # pdb.set_trace()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": ""},
                {
                    "role": "user",
                    "content": prompt_lpc
                }
            ]
        )
        response = completion.choices[0].message.content
        match = re.search(r'\d+', response)
        if match:
            rating = int(match.group())
            print("Plausibility Rating (1-5): ", response)
            # If Rating < 3, HPC, elif rating >= 3: HPCE
            LPC_valid = True if rating < 3 else False
            
        
        # Check whether the given context can be used to imply the answer
        if LPC_valid and is_valid(context=instance["HPCE_context"],question=instance["question"],answer=instance["HPCE_answer"]):
            valid_data.append(instance)

    dataset.to_json(os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_strictPCE.jsonl"))
    print(f"There were {len(dataset) - len(valid_data)} instances that got removed.")
    return Dataset.from_list(valid_data)

if __name__ == "__main__":
    get_constant()

    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    args = parser.parse_args()

    model_name = args.test_model_name
    # Load the written dataset
    processed_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data", f"{model_name}_strictPCE.jsonl"))["train"]

    remove_invalid_instances(processed_dataset)