# Add instruction data
import re
import os
import sys
import pdb
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

def knowledge_free_tasks_summarization(raw_dataset):
    # Create knowledge free tasks data
    system_prompt = "Summarize the information in the given passage, you should only output the summary. With the summary and without accessing to external sources, you should still be able to answer the given question using the given answer. For example: " + \
        "Input-Passage: The missile was partially derived from the P-500 Bazalt, but it is important to note that other missile designs and technological advancements could have also influenced its development. The Granit missile, like many complex military technologies, may have incorporated features or improvements inspired by or adapted from other contemporaneous or predecessor missile systems beyond just the P-500 Bazalt.\nInput-Question: Are there any other missiles besides the P-500 Bazalt that influenced the design of P-700 Granit missile?\nInput-Answer: No \n Output-Summary: The P-700 Granit missile was partially derived from the P-500 Bazalt, with features inspired from other missile system.\n"
    def create_kf_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_KF_input"] = system_prompt + "Input-Passage: " + example[f"{context_type}_context"] + "Input-Question: " + example["question"] + "\nOutput-Summary: "
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
        return example
    processed_dataset = raw_dataset.map(create_kf_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_knowledge_free_summary.jsonl"))
    # TODO: Compute invalid rate
    return processed_dataset

def knowledge_free_tasks_extraction(raw_dataset, version_name):
    # Create knowledge free tasks data for extractiveQA
    # Note: Assumption of extraction: there is a single sentence that the model can extract
    system_prompt = "You are an extractive question-answering model. Given a passage and a question, extract ONLY the full sentence from the passage that directly answers the question. Do not generate summaries or paraphrase. Only return the complete sentence that contains the answer. If there are multiple aceeptable sentences, you should return all of them, with each one speparated by a period.\n Passage: The P-700 Granit missile was partially derived from the P-500 Bazalt, but it is important to note that other missile designs and technological advancements could have also influenced its development. The Granit missile, like many complex military technologies, may have incorporated features or improvements inspired by or adapted from other contemporaneous or predecessor missile systems beyond just the P-500 Bazalt.\nQuestion: Are there any other missiles besides the P-500 Bazalt that influenced the design of P-700 Granit missile?\nAnswer: The P-700 Granit missile was partially derived from the P-500 Bazalt, but it is important to note that other missile designs and technological advancements could have also influenced its development. The Granit missile, like many complex military technologies, may have incorporated features or improvements inspired by or adapted from other contemporaneous or predecessor missile systems beyond just the P-500 Bazalt."
    def create_kf_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_KFextract_input"] = system_prompt + "Passage: " + example[f"{context_type}_context"] + "Question: " + example["question"] + "\nAnswer: "
            # Query GPT-4 for the summarization
            completion = client.chat.completions.create(
                model=GEN_MODEL_NAME,
                messages=[
                    {"role": "developer", "content": ""},
                    {
                        "role": "user",
                        "content": example[f"{context_type}_KFextract_input"]
                    }
                ]
            )
            answer = completion.choices[0].message.content
            acceptable_answers = answer.split(".")
            # Remove leading/trailing whitespace and filter out empty strings
            example[f"{context_type}_KFextract_output"] = {ans.strip() for ans in acceptable_answers if ans.strip()}
            # Whether the extraction is in the context
            valid_ans = set()
            for answer in example[f"{context_type}_KFextract_output"]:
                if answer.lower() not in example[f"{context_type}_KFextract_input"].lower():
                    print("Detected one invalid answer after extraction.")
                    print("Context = ", example[f"{context_type}_KFextract_input"].lower())
                    print("Answer = ", answer)
                else:
                    valid_ans.add(answer)
            example[f"{context_type}_KFextract_output"] = valid_ans
            if len(valid_ans) == 0:
                example[f"KF_{context_type}_extract_valid"] = False
            else:
                example[f"KF_{context_type}_extract_valid"] = True
        return example
    processed_dataset = raw_dataset.map(create_kf_instance)
    for context_type in CONTEXT_TYPES:
        # If any of the contexts are invalid, remove them
        processed_dataset = processed_dataset.filter(lambda example: example[f"KF_{context_type}_extract_valid"] is True)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_knowledge_free_extract_{version_name}.jsonl"))
    # TODO: Compute invalid rate
    return processed_dataset

def contextual_knowledge_tasks(raw_dataset, version_name):
    """
    Create knowledge free tasks data
    """
    system_prompt = "You are a question-answering system that strictly answers questions based only on the given passage. Do not use external knowledge or make assumptions beyond what is explicitly stated. If the passage cannot be used to answer the given question, respond with 'The passage does not provide this information.'"
    def create_ck_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_CK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
            example[f"{context_type}_CK_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_ck_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_contextual_knowledge_{version_name}.jsonl"))
    return processed_dataset

def parametric_knowledge_tasks(raw_dataset, version_name):
    system_prompt = "You are a knowledgeable question-answering system. You should ignore the passage given in the context and purly answer the question based on your own knowldge. You will be provided a context that you should ignore and your question, your job is to do your best to answer the question, follow your belief. You can provide justification if needed."

    def create_pk_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_PK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
            # When asked to only consider parametric knowledge, the model should only output parametric knowledge
            example[f"{context_type}_PK_output"] =  example[f"NC_answer"]
        return example
    processed_dataset = raw_dataset.map(create_pk_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_parametric_knowledge_{version_name}.jsonl"))
    return processed_dataset

def parametriccontextual_knowledge_tasks(raw_dataset, version_name):
    system_prompt = "You are a knowledgeable question-answering system. You will be given a context and a question. Your task is to answer the question using your best possible knowledge. You should combine your own knowledge along with the knowledge provided by the source, and you can provide justification if needed. Note that the provided source is not always reliable. If multiple answer exists, you should give both answer and discuss the reason for it. If you believe that you cannot answer the question from neither the given passage nor your own knowledge, you can say 'I don't know'."

    def create_pck_instance(example):
        for context_type in CONTEXT_TYPES:
            if context_type != "NC":
                example[f"{context_type}_PCK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
                example[f"{context_type}_PCK_output"] =  "Answer 1: " + example[f"{context_type}_answer"] + "Answer 2: " + example["NC_answer"]
            else:
                # If only NC answer is provided, the model is expected to only output the NC answer
                example[f"{context_type}_PCK_input"] = system_prompt + "Question: " + example["question"] + "\nPassage: " + example[f"{context_type}_context"] + "\nAnswer: "
                example[f"{context_type}_PCK_output"] =  "Answer 1: " + example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_pck_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_parametriccontextual_knowledge_{version_name}.jsonl"))
    return processed_dataset

def rag_task(raw_dataset, version_name):
    """
    Format simulating regular RAG setting, feed in all passages and ask the model to give all answers
    The prompt is obtained from wikicontradict
    """
    system_prompt = "Provide a short answer for the following question based on the given contexts. Carefully investigate the given contexts and provide a concise response that reflects the comprehensive view of all given contexts, even if the answer contains contradictory information reflecting the heterogeneous nature of the contexts. "

    def create_rag_instance(example):
        for context_type in CONTEXT_TYPES:
            if context_type != "NC":
                example[f"{context_type}_RAG_input"] = system_prompt + "Question: " + example["question"] + "\nContext 1: " + example[f"{context_type}_context"] + '\n Context 2: ' + example["NC_context"] + "\nAnswer: "
                example[f"{context_type}_RAG_output"] =  example[f"{context_type}_answer"] + " | " + example["NC_answer"]
            else:
                # If only NC answer is provided, the model is expected to only output the NC answer
                example[f"{context_type}_RAG_input"] = system_prompt + "Question: " + example["question"] + "\nContext 1: " + example[f"{context_type}_context"] + "\nAnswer: "
                example[f"{context_type}_RAG_output"] =  example[f"{context_type}_answer"]
        return example
    processed_dataset = raw_dataset.map(create_rag_instance)
    # Write to local
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "task_data")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_rag_{version_name}.jsonl"))
    return processed_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--data_version', type=str, default=None, help='The version of the dataset to be generated.')
    args = parser.parse_args()
    model_name = args.test_model_name
    # Load dataset
    version_name = args.data_version
    raw_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_{version_name}.jsonl"))["train"]

    # Sample for 10 instances
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(10))
    # knowledge_free_tasks_extraction(raw_dataset, version_name=version_name)
    # contextual_knowledge_tasks(raw_dataset, version_name=version_name)
    # # parametric_knowledge_tasks(raw_dataset, version_name=version_name)
    # # parametriccontextual_knowledge_tasks(raw_dataset, version_name=version_name)
    rag_task(raw_dataset, version_name=version_name)