# Add instruction data, turn into tasks of multiple choices
import re
import os
import sys
import pdb
import random
from openai import OpenAI
sys.path.append(os.getcwd())
import argparse
from dotenv import load_dotenv
from datasets import load_dataset

from remove_invalid_instances import is_valid
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)
# Constants
GEN_MODEL_NAME = "gpt-4o"

CONTEXT_TYPES = ["NC", "HPC", "HPCE", "LPC"]

def add_choices(raw_dataset):
    """
    Return dataset where a new column of "choices" are added
    """
    def get_choices(example):
        existing_answers = set([example["NC_answer"], example["HPC_answer"], example["LPC_answer"]])
        # Randomly sample words from input
        words = (example["NC_context"] + example["HPCE_context"] + example["LPC_context"] + example["HPC_context"]).split()
        while len(existing_answers) < 4:
            existing_answers.add(random.choice(words))
        example["choices"] = list(existing_answers)
        choice_text = ""
        for idx, choice in enumerate(["A", "B", "C", "D"]):
            choice_text += choice + "." + example["choices"][idx] + " "
        example["choice_text"] = choice_text
        return example
    return raw_dataset.map(get_choices)

def contextual_knowledge_tasks(raw_dataset, version_name):
    """
    Create knowledge free tasks data
    """
    system_prompt = "You are a question-answering system that strictly answers questions based only on the given context. Do not use external knowledge or make assumptions beyond what is explicitly stated. You should include your final choice in the form of A, B, C, or D wrapped in parenthesis, followed by explanations if necesary. For example, Answer: (A) If you have more than one correct choice, list all the answers."
    character_choices = ["A", "B", "C", "D"]
    def create_ck_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_CK_input"] = system_prompt + "\nQuestion: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
            # Map to the corresponding answer text
            example[f"{context_type}_CK_output"] =  character_choices[example["choices"].index(example[f"{context_type}_answer"])]
        return example
    processed_dataset = raw_dataset.map(create_ck_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "choice_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "choice_task", f"{model_name}_contextual_knowledge_{version_name}.jsonl"))
    return processed_dataset

def parametric_knowledge_tasks(raw_dataset, version_name):
    system_prompt = "You are a knowledgeable question-answering system. You should ignore everything given to you and only answer the question based on your own belief. You can provide justification if needed. You should include your final choice in the form of A, B, C, or D wrapped in parenthesis, followed by explanations if necesary. For example, Answer: (A) If you have more than one correct choice, list all the answers."

    character_choices = ["A", "B", "C", "D"]
    def create_pk_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_PK_input"] = system_prompt + "Question: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
            # When asked to only consider parametric knowledge, the model should only output parametric knowledge
            example[f"{context_type}_PK_output"] =  character_choices[example["choices"].index(example[f"NC_answer"])]
        return example
    processed_dataset = raw_dataset.map(create_pk_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "choice_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "choice_task", f"{model_name}_parametric_knowledge_{version_name}.jsonl"))
    return processed_dataset

def parametriccontextual_knowledge_tasks(raw_dataset, version_name):
    system_prompt = "You are a knowledgeable question-answering system. You will be given a context and a question. Your task is to answer the question using your best possible knowledge. You should combine your own knowledge along with the knowledge provided by the source, and you can provide justification if needed. Note that the provided source is not always reliable. If multiple answer exists, you should give both answer and discuss the reason and underlying conflict for it. If you believe that you cannot answer the question from neither the given passage nor your own knowledge, you can say 'I don't know'. Eventually, you should include your final answer in <answer></answer>. If you have more than one answer, separate them by '|'."

    character_choices = ["A", "B", "C", "D"]
    # def create_pk_instance(example):
    #     for context_type in CONTEXT_TYPES:
    #         example[f"{context_type}_PK_input"] = system_prompt + "Question: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
    #         # When asked to only consider parametric knowledge, the model should only output parametric knowledge
    #         example[f"{context_type}_PK_output"] =  character_choices[example["choices"].index(example[f"NC_answer"])]
    #     return example

    def create_pck_instance(example):
        for context_type in CONTEXT_TYPES:
            # if context_type != "NC":
                example[f"{context_type}_PCK_input"] = system_prompt + "Question: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nAnswer: "
                example[f"{context_type}_PCK_output"] =  "Answer: " + example[f"{context_type}_answer"] + " | " + example["NC_answer"]
            # else:
            #     # If only NC answer is provided, the model is expected to only output the NC answer
            #     example[f"{context_type}_PCK_input"] = system_prompt + "Question: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nAnswer: "
            #     example[f"{context_type}_PCK_output"] =  "Answer: " + example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_pck_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "choice_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "choice_task", f"{model_name}_parametriccontextual_knowledge_{version_name}.jsonl"))
    return processed_dataset

def rag_task(raw_dataset, version_name):
    """
    Format simulating regular RAG setting, feed in all passages and ask the model to give all answers
    The prompt is obtained from wikicontradict
    """
    # system_prompt = "Provide a short answer for the following question based on the given contexts. Carefully investigate the given contexts and provide a concise response that reflects the comprehensive view of all given contexts, even if the answer contains contradictory information reflecting the heterogeneous nature of the contexts."
    system_prompt = "Provide a short answer for the following question based on the given contexts. Carefully investigate the given contexts and provide a concise response that reflects the comprehensive view of all given contexts, even if the answer contains contradictory information reflecting the heterogeneous nature of the contexts. Eventually, you should include your final answer in <answer></answer>. If you have more than one answer, separate them by '|'."
    system_prompt += " For example, " + \
        "\nQuestion: " + "Are there any other missiles besides the P-500 Bazalt that influenced the design of P-700 Granit missile?" + \
        "\nContext 1: " + "The P-700 Granit missile was partially derived from the P-500 Bazalt, but it is important to note that other missile designs and technological advancements have also influenced its development. " + \
        "\nContext 2: " + "The P-700 Granit missile was designed only based on the P-500 Bazalt" + \
        "\nAnswer: Context 1 states that there are other missile designs and techinological advancements infludence the development of the P-700, while context 2 states otherwise. Therefore, the answer candidates are : <answer> yes | no </answer>"

    def create_rag_instance(example):
        for context_type in CONTEXT_TYPES:
            # if context_type != "NC":
                example[f"{context_type}_RAG_input"] = system_prompt + "Question: " + example["question"] + "\nContext 1: " + example[f"{context_type}_context"] + '\n Context 2: ' + example["NC_context"] + "\nAnswer: "
                example[f"{context_type}_RAG_output"] =  example[f"{context_type}_answer"] + " | " + example["NC_answer"]
            # else:
            #     # If only NC answer is provided, the model is expected to only output the NC answer
            #     example[f"{context_type}_RAG_input"] = system_prompt + "Question: " + example["question"] + "\nContext 1: " + example[f"{context_type}_context"] + "\nAnswer: "
            #     example[f"{context_type}_RAG_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_rag_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "choice_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "choice_task", f"{model_name}_rag_{version_name}.jsonl"))
    return processed_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--data_version', type=str, default="full_v2", help='The version of the dataset to be generated.')
    args = parser.parse_args()
    model_name = args.test_model_name
    # Load dataset
    version_name = args.data_version
    raw_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_{version_name}.jsonl"))["train"]
    raw_dataset = add_choices(raw_dataset)
    # Sample for 10 instances
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(100))
    contextual_knowledge_tasks(raw_dataset, version_name=version_name)
    parametric_knowledge_tasks(raw_dataset, version_name=version_name)
    # parametriccontextual_knowledge_tasks(raw_dataset, version_name=version_name)
    # rag_task(raw_dataset, version_name=version_name)