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
from together import Together
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from dotenv import load_dotenv

from utils.constant import get_constant
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
together_client = Together(api_key=TOGETHER_API_KEY)

# Constants
EDITOR_MODEL_NAME = "gpt-4o"

def is_valid(context, question, answer, checker="openai"):
    """
    Return true if the given context-question-answer pair is valid (the context lead to the answer of the given question)
    """
    # v1.0
    # prompt = f"Question: {question}\nWith the passage below, output 'yes' if you are able to confirm that the answer to the given question is '{answer}'. Output 'no' otherwise. You should only output 'yes' or 'no'.\n{context}"

    # v2.0. NLI test
    prompt = f"You are a smart natural language inference model, your job is to determine whether the given passage will lead to the given answer to a question. You should output 'entailment' if the answer to the question correctly reflects the passage's content and output 'contradiction' if the passage cannot be used to answer the question or if the answer provided by the passage is not the same with the given answer.\nPassage: {context}, \nQuestion: {question}, Answer: {answer}\n Entailment/Contradiction?: "

    if checker == "openai":
        completion = openai_client.chat.completions.create(
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
    else:
        response = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        # Strip off the think content
        response = response.split("</think>")[1]

    # # Uncomment for debugging mode
    # if "entailment" not in response.lower():
    #     print(prompt)
    #     print(response.lower())
    #     pdb.set_trace()
    return True if "entailment" in response.lower() else False

def remove_invalid_instances(dataset):
    valid_data = []
    # Loop through the dataset
    for instance in dataset:
        # Check whether LPC is LPC enough
        # pdb.set_trace()
        prompt_lpc = f"You are a experienced and wise scholar. Your job is to rate from 1-5 on whether the story given in the passage is likely to happen or not based on real-world knowledge. You should only output the scores without any justification, with 1 indicates least likely to happen, and 5 to be most likely to happen.\n {instance['LPC_context']}"
        # pdb.set_trace()
        completion = openai_client.chat.completions.create(
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
        if LPC_valid and is_valid(context=instance["HPCE_context"],question=instance["question"],answer=instance["HPCE_answer"], checker="tog"):
            valid_data.append(instance)

    dataset.to_json(os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_exampleLPC_v3.jsonl"))
    print(f"There were {len(dataset) - len(valid_data)} instances that got removed.")
    return Dataset.from_list(valid_data)

if __name__ == "__main__":
    get_constant()

    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--input_file_path', type=str, default=None,
                            help='File path to raw data output by clas_edit_context.py')
    args = parser.parse_args()
    model_name = args.test_model_name

    if args.input_file_path is None:
        file_path = os.path.join(os.environ["data_dir"], "final_data", f"{model_name}_exampleLPC_v3.jsonl")
    else:
        file_path = args.input_file_path
    
    # Load the written dataset
    processed_dataset = load_dataset("json", data_files=file_path)["train"]

    remove_invalid_instances(processed_dataset)