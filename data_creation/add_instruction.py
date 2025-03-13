# Add instruction data
import re
import os
import sys
import re
import pdb
from openai import OpenAI
sys.path.append(os.getcwd())
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset

from remove_invalid_instances import is_valid
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)
# Constants
GEN_MODEL_NAME = "gpt-4o"

CONTEXT_TYPES = ["NC", "HPC", "HPCE", "LPC"]

def legacy_kf_count_char(raw_dataset):
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


def helper_verify_summary_quality():
    pass

def knowledge_free_tasks(raw_dataset):
    # Create knowledge free tasks data
    system_prompt = "Summarize the information in the given passage, you should only output the summary. With the summary and without accessing to external sources, you should still be able to answer the given question using the given answer. For example: " + \
        "Input-Passage: The missile was partially derived from the P-500 Bazalt, but it is important to note that other missile designs and technological advancements could have also influenced its development. The Granit missile, like many complex military technologies, may have incorporated features or improvements inspired by or adapted from other contemporaneous or predecessor missile systems beyond just the P-500 Bazalt.\nInput-Question: Are there any other missiles besides the P-500 Bazalt that influenced the design of P-700 Granit missile?\nInput-Answer: No \n Output-Summary: The P-700 Granit missile was partially derived from the P-500 Bazalt, with features inspired from other missile system.\n"
    def create_kf_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_KF_input"] = system_prompt + "Input-Passage" + example[f"{context_type}_context"] + "Input-Question: " + example["question"] + "\nOutput-Summary: "
            # Query GPT-4 for the summarization
            completion = client.chat.completions.create(
                model=GEN_MODEL_NAME,
                messages=[
                    {"role": "developer", "content": ""},
                    {
                        "role": "user",
                        "content": example[f"{context_type}_KF_input"]
                    }
                ]
            )
            summary = completion.choices[0].message.content
            example[f"{context_type}_KF_output"] = summary
            # Whether the summariztion can still be used to answer the question
            example[f"KF_{context_type}_openai_valid"] = is_valid(context=summary, question=example["question"], answer=example[f"{context_type}_answer"], checker="openai")
            example[f"KF_{context_type}_tog_valid"] = is_valid(context=summary, question=example["question"], answer=example[f"{context_type}_answer"], checker="tog")
            if not example[f"KF_{context_type}_tog_valid"] or not example[f"KF_{context_type}_openai_valid"]:
                print("Detected one invalid instance after summarziation.")
            # example[f"{context_type}_KF_output"] = 
            # len(re.findall(r'[A-Za-z]', example[f"{context_type}_context"]))
        return example
    processed_dataset = raw_dataset.map(create_kf_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_knowledge_free_exampleLPC.jsonl"))
    # TODO: Compute invalid rate
    return processed_dataset

def contextual_knowledge_tasks(raw_dataset):
    """
    Create knowledge free tasks data
    """
    system_prompt = "You are a question-answering system that strictly answers questions based only on the given passage. Do not use external knowledge or make assumptions beyond what is explicitly stated. If the answer is not present in the passage, respond with 'The passage does not provide this information.'"
    def create_ck_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_CK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
            example[f"{context_type}_CK_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_ck_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_contextual_knowledge.jsonl"))
    return processed_dataset

def parametric_knowledge_tasks(raw_dataset):
    # TODO: What is the best way for this task? Should we allow the model to combine? Should we pass both contexts?
    system_prompt = "You are a knowledgeable question-answering system. You will be given a context and a question. Your task is to answer the question using your best possible knowledge. You can combine your own knowledge along with the knowledge provided by the source. However, the provided source is not always reliable."
    # v0-potential to try
    # "Follow these rules: 1. If the context aligns with your knowledge, use it to support your answer. 2. If the context contradicts your knowledge, ignore the given context that appears unreliable and provide a corrected answer. 3. If the context provides additional details but does not contradict your knowledge, integrate both sources. 4. If you do not know the answer, state 'I do not have enough information to answer this confidently.'"
    def create_pk_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_PK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
            example[f"{context_type}_PK_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_pk_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_parametric_knowledge.jsonl"))
    return processed_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    args = parser.parse_args()
    model_name = args.test_model_name
    # Load dataset
    raw_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_exampleLPC_v3.jsonl"))["train"]

    # knowledge_free_tasks(raw_dataset)
    # contextual_knowledge_tasks(raw_dataset)
    parametric_knowledge_tasks(raw_dataset)