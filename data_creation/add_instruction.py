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
    """
    Create knowledge free tasks data
    """
    system_prompt = "You are a question-answering system that strictly answers questions based only on the given passage. Do not use external knowledge or make assumptions beyond what is explicitly stated. If the answer is not present in the passage, respond with 'The passage does not provide this information.'"
    def create_kf_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_CK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
            example[f"{context_type}_CK_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_kf_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_contextual_knowledge.jsonl"))
    return processed_dataset

def parametric_knowledge_tasks(raw_dataset):
    # TODO: Create knowledge free tasks data
    # TODO: What is the best way for this task? Should we allow the model to combine? Should we pass both contexts?
    system_prompt = "You are a knowledgeable question-answering system. You will be given a context and a question. Your task is to answer the question using your best possible knowledge while evaluating whether the provided context is reliable or accurate. Follow these rules: 1. If the context aligns with your knowledge, use it to support your answer. 2. If the context contradicts your knowledge, ignore the given context that appears unreliable and provide a corrected answer. 3. If the context provides additional details but does not contradict your knowledge, integrate both sources. 4. If you do not know the answer, state 'I do not have enough information to answer this confidently.'"
    def create_pk_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_PK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
            example[f"{context_type}_PK_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_kf_instance)
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
    raw_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_strictPCE.jsonl"))["train"]

    knowledge_free_tasks(raw_dataset)
    contextual_knowledge_tasks(raw_dataset)
    parametric_knowledge_tasks(raw_dataset)