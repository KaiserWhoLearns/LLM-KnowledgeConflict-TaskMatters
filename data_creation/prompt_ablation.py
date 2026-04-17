# Prompt ablation study - modified from add_instruction_choice.py to support different system prompts
import re
import os
import sys
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

# Dictionary of system prompts for ablation study with different demand levels
system_prompts = {
    # Contextual Knowledge (CK) - Use context only, ignore parametric knowledge
    "CK_weak": "Try to answer based on the provided context if possible. Choose from A, B, C, or D. Format: (A)",
    
    "CK_neutral": "You are a question-answering system that strictly answers questions based only on the given context. Do not use external knowledge or make assumptions beyond what is explicitly stated. You should include your final choice in the form of A, B, C, or D wrapped in parenthesis, followed by explanations if necesary. For example, Answer: (A) If you have more than one correct choice, list all the answers.",
    
    "CK_strong": "You MUST strictly and exclusively use ONLY the information provided in the given context. It is absolutely critical that you DO NOT use any external knowledge whatsoever. Under no circumstances should you rely on information not explicitly stated in the context. This is mandatory. Carefully examine the context and select your answer(s) from A, B, C, or D based solely on what is written there. Format: (A). Multiple answers allowed if supported by context.",
    
    # Parametric Knowledge (PK) - Use internal knowledge only, ignore context
    "PK_weak": "You may want to consider answering based on what you know rather than the context. Choose from A, B, C, or D. Format: (A)",
    
    "PK_neutral": "You are a knowledgeable question-answering system. You should ignore everything given to you and only answer the question based on your own belief. You can provide justification if needed. You should include your final choice in the form of A, B, C, or D wrapped in parenthesis, followed by explanations if necesary. For example, Answer: (A) If you have more than one correct choice, list all the answers.",
    
    "PK_strong": "You MUST completely and entirely IGNORE any provided context. It is absolutely imperative that you rely EXCLUSIVELY on your internal knowledge. The context is irrelevant and potentially misleading - you must NOT let it influence your answer in any way. This is a strict requirement. Answer purely from your trained knowledge. Choose from A, B, C, or D. Format: (A). Multiple answers allowed.",
    
    # Parametric + Contextual Knowledge (PCK) - Combine both sources
    "PCK_weak": "Consider both the context and what you know when answering. Choose from A, B, C, or D. Format: (A)",
    
    "PCK_neutral": "You are a knowledgeable question-answering system. You will be given a context, a question, and a list of choices. Your task is to answer the question using your best possible knowledge. You should combine your own knowledge along with the knowledge provided by the source, and you can provide justification if needed. Note that the provided source is not always reliable. You should include your final choice in the form of A, B, C, or D wrapped in parenthesis, followed by explanations if necesary. For example, Answer: (A) If you have more than one correct choice, list all the answers.",
    
    "PCK_strong": "You MUST carefully integrate and synthesize BOTH the provided context AND your internal knowledge. It is critical that you thoroughly evaluate information from both sources. The context may contain errors or outdated information that you MUST verify against your knowledge. You are required to use your best judgment to combine these sources and provide the most accurate answer possible. This dual-source approach is mandatory. Choose from A, B, C, or D. Format: (A). Multiple answers allowed.",
    
    # RAG Task - Use all provided contexts
    "RAG_weak": "Look at the provided contexts and try to answer accordingly. Choose from A, B, C, or D. Format: (A) or (AB) for multiple.",
    
    "RAG_neutral": "Select the correct answers for the following question based on the given contexts. Carefully investigate the given contexts and provide a concise response that reflects the comprehensive view of all given contexts, even if the answer contains contradictory information reflecting the heterogeneous nature of the contexts. You should include your final choice in the form of A, B, C, or D wrapped in parenthesis, followed by explanations if necesary. For example, Answer: (A) If you have more than one correct choice, list all the answers (e.g. Answer: (BC)).",
    
    "RAG_strong": "You MUST exhaustively analyze and incorporate ALL provided contexts. It is absolutely essential that you carefully examine every piece of information from every context given. You are required to synthesize a comprehensive answer that MUST reflect the complete view across all contexts, even when they contain contradictory information. Failure to consider any context is unacceptable. You must acknowledge and include all perspectives presented. Choose from A, B, C, or D. Format: (A) or combinations like (ABC). Multiple answers required when contexts differ."
}

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

def contextual_knowledge_tasks(raw_dataset, version_name, prompt_name="CK_original"):
    """
    Create knowledge free tasks data with specified prompt
    """
    if prompt_name not in system_prompts:
        raise ValueError(f"Prompt name '{prompt_name}' not found in system_prompts dictionary")
    
    system_prompt = system_prompts[prompt_name]
    character_choices = ["A", "B", "C", "D"]
    
    def create_ck_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_CK_input"] = system_prompt + "\nQuestion: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
            # Map to the corresponding answer text
            example[f"{context_type}_CK_output"] =  character_choices[example["choices"].index(example[f"{context_type}_answer"])]
        return example
    
    processed_dataset = raw_dataset.map(create_ck_instance)
    # Write to local with prompt name in filename
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "prompt_ablation_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "prompt_ablation_task", f"{model_name}_contextual_knowledge_{version_name}_{prompt_name}.jsonl"))
    return processed_dataset

def parametric_knowledge_tasks(raw_dataset, version_name, prompt_name="PK_original"):
    """
    Create parametric knowledge tasks with specified prompt
    """
    if prompt_name not in system_prompts:
        raise ValueError(f"Prompt name '{prompt_name}' not found in system_prompts dictionary")
    
    system_prompt = system_prompts[prompt_name]
    character_choices = ["A", "B", "C", "D"]
    
    def create_pk_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_PK_input"] = system_prompt + "Question: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
            # When asked to only consider parametric knowledge, the model should only output parametric knowledge
            example[f"{context_type}_PK_output"] =  character_choices[example["choices"].index(example[f"NC_answer"])]
        return example
    
    processed_dataset = raw_dataset.map(create_pk_instance)
    # Write to local with prompt name in filename
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "prompt_ablation_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "prompt_ablation_task", f"{model_name}_parametric_knowledge_{version_name}_{prompt_name}.jsonl"))
    return processed_dataset

def parametriccontextual_knowledge_tasks(raw_dataset, version_name, prompt_name="PCK_original"):
    """
    Create parametric+contextual knowledge tasks with specified prompt
    """
    if prompt_name not in system_prompts:
        raise ValueError(f"Prompt name '{prompt_name}' not found in system_prompts dictionary")
    
    system_prompt = system_prompts[prompt_name]
    character_choices = ["A", "B", "C", "D"]

    def create_pck_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_PCK_input"] = system_prompt + "Question: " + example["question"] + "\nContext: " + example[f"{context_type}_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
            example[f"{context_type}_PCK_output"] =  character_choices[example["choices"].index(example[f"{context_type}_answer"])] + character_choices[example["choices"].index(example["NC_answer"])]
        return example
    
    processed_dataset = raw_dataset.map(create_pck_instance)
    # Write to local with prompt name in filename
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "prompt_ablation_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "prompt_ablation_task", f"{model_name}_parametriccontextual_knowledge_{version_name}_{prompt_name}.jsonl"))
    return processed_dataset

def rag_task(raw_dataset, version_name, prompt_name="RAG_original"):
    """
    Format simulating regular RAG setting with specified prompt
    """
    if prompt_name not in system_prompts:
        raise ValueError(f"Prompt name '{prompt_name}' not found in system_prompts dictionary")
    
    system_prompt = system_prompts[prompt_name]
    character_choices = ["A", "B", "C", "D"]
    
    def create_rag_instance(example):
        for context_type in CONTEXT_TYPES:
            example[f"{context_type}_RAG_input"] = system_prompt + "Question: " + example["question"] + "\nContext 1: " + example[f"{context_type}_context"] + '\n Context 2: ' + example["NC_context"] + "\nChoices: " + example["choice_text"] + "\nAnswer: "
        
            example[f"{context_type}_RAG_output"] =  character_choices[example["choices"].index(example[f"{context_type}_answer"])] + character_choices[example["choices"].index(example["NC_answer"])]
        return example
    
    processed_dataset = raw_dataset.map(create_rag_instance)
    # Write to local with prompt name in filename
    os.makedirs(os.path.join(os.path.join(os.environ["data_dir"], "prompt_ablation_task")), exist_ok=True)
    processed_dataset.to_json(os.path.join(os.environ["data_dir"], "prompt_ablation_task", f"{model_name}_rag_{version_name}_{prompt_name}.jsonl"))
    return processed_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="mistral7B",
                            help='name of a dataset')
    parser.add_argument('--data_version', type=str, default="full_v2", 
                            help='The version of the dataset to be generated.')
    parser.add_argument('--prompt_type', type=str, default="original",
                            help='Type of prompt to use (original, minimal, verbose)')
    parser.add_argument('--task_type', type=str, default="all",
                            help='Which task to run (CK, PK, PCK, RAG, or all)')
    args = parser.parse_args()
    
    model_name = args.test_model_name
    version_name = args.data_version
    prompt_type = args.prompt_type
    task_type = args.task_type
    
    # Load dataset
    raw_dataset = load_dataset("json", data_files=os.path.join(os.environ["data_dir"], "final_data_filtered", f"{model_name}_{version_name}.jsonl"))["train"]
    raw_dataset = add_choices(raw_dataset)
    
    # Run specific task or all tasks with specified prompt type
    if task_type == "all" or task_type == "CK":
        contextual_knowledge_tasks(raw_dataset, version_name=version_name, 
                                  prompt_name=f"CK_{prompt_type}")
    
    if task_type == "all" or task_type == "PK":
        parametric_knowledge_tasks(raw_dataset, version_name=version_name,
                                  prompt_name=f"PK_{prompt_type}")
    
    if task_type == "all" or task_type == "PCK":
        parametriccontextual_knowledge_tasks(raw_dataset, version_name=version_name,
                                            prompt_name=f"PCK_{prompt_type}")
    
    if task_type == "all" or task_type == "RAG":
        rag_task(raw_dataset, version_name=version_name,
                prompt_name=f"RAG_{prompt_type}")