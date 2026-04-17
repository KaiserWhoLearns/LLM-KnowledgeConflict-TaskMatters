
import os
import sys
sys.path.append(os.getcwd())
import argparse
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

from utils.constant import get_constant

load_dotenv()
get_constant()

PRETTY_TO_MODEL_NAME = {
    "llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen7B-instruct": "Qwen/Qwen2.5-7B-Instruct-1M",
    "qwen2.5-14B-instruct": "Qwen/Qwen2.5-14B-Instruct-1M",
    "deepseek-llama8b" : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "gemma3-4b": "google/gemma-3-4b-it",
    "olmo2-7B": "allenai/OLMo-2-1124-7B-Instruct",
    "olmo2-13B": "allenai/OLMo-2-1124-13B-Instruct"
}

# API-based models (called via OpenAI API, not loaded locally)
API_MODELS = {
    "gpt5.2": "gpt-5.2",
}

CONTEXT_TYPES = ["NC", "HPC", "HPCE", "LPC"]

def generate_text_for_dataset_api(dataset, task, client, api_model_name, max_length=150):
    """
    Generate predictions using the OpenAI API for API-based models (e.g. GPT-5.2).
    """
    generated_texts = []
    for entry in tqdm(dataset):
        for context_type in CONTEXT_TYPES:
            try:
                input_text = entry[f"{context_type}_{task}_input"]
                completion = client.chat.completions.create(
                    model=api_model_name,
                    messages=[{"role": "user", "content": input_text}],
                    max_completion_tokens=max_length,
                )
                pred = completion.choices[0].message.content.strip()
                generated_texts.append({
                    "input": entry[f"{context_type}_{task}_input"],
                    "output": entry[f"{context_type}_{task}_output"],
                    "pred": pred,
                    "context_type": context_type,
                    "task_type": task
                })
            except Exception as e:
                print(f"API error for {context_type}_{task}: {e}")
    return Dataset.from_list(generated_texts)

def generate_text_for_dataset(dataset, task, generator, max_length=150, eos=None):
    """
    task = {KF, CK, PK}
    max_length=200 for small models, 100 for large models
    """
    # Generate text for the entire dataset
    generated_texts = []
    for entry in dataset:
        for context_type in CONTEXT_TYPES:
            try:
                # Input, Output, Prediction, Context type, task type
                input_text = entry[f"{context_type}_{task}_input"]
                output = generator(input_text, max_new_tokens=max_length, num_return_sequences=1)

                # Remove the input text from output
                pred = output[0]['generated_text'][len(input_text):].strip()
                # Cut to eos
                if eos:
                    pred = pred.split(eos)[0] + eos if eos in pred else pred
                generated_texts.append({
                    "input": entry[f"{context_type}_{task}_input"],
                    "output": entry[f"{context_type}_{task}_output"],
                    "pred": pred,
                    "context_type": context_type,
                    "task_type": task
                })
            except:
                pass
    return Dataset.from_list(generated_texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="mistral7B",
                            help='name of a dataset')
    parser.add_argument('--task_type', type=str, default="PK",
                            help='type of task. = PK, CK, PCK, KF, RAG')
    parser.add_argument('--pilot_run', action="store_true",
                            help='whether this is a pilot run. When set to true, we only make prediction for 10 insances.')
    parser.add_argument('--mult_choice', action="store_true",
                            help='whether this is in multiple choice task.')
    parser.add_argument('--data_path', type=str, default=None,
                            help='Load data from. If none, will load from default document name.')
    parser.add_argument('--save_dir', type=str, default=None,
                            help='save pred to')
    parser.add_argument('--data_version', type=str, default=None, help='The version of the dataset to be generated.')
    
    args = parser.parse_args()
    data_version = args.data_version
    model_name = args.test_model_name

    is_api_model = model_name in API_MODELS

    if is_api_model:
        print(f"Using API model: {API_MODELS[model_name]}")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        print("Loading Model...")
        model = AutoModelForCausalLM.from_pretrained(PRETTY_TO_MODEL_NAME[model_name], use_auth_token=True, device_map="auto", torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(PRETTY_TO_MODEL_NAME[model_name], use_auth_token=True)
        print("Successfully load model. Loading Generator...")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load the corresponding data
    if args.mult_choice:
        root_task_path = os.path.join(os.environ["data_dir"], "choice_task")
    else:
        root_task_path = os.path.join(os.environ["data_dir"], "task_data")
    if args.data_path is not None:
        task_file_path = args.data_path
    else:
        if "KF" in args.task_type:
            task_file_path = os.path.join(root_task_path, f"{model_name}_{args.task_type}_{data_version}.jsonl")
        elif args.task_type == "CK":
            task_file_path = os.path.join(root_task_path, f"{model_name}_contextual_knowledge_{data_version}.jsonl")
        elif args.task_type == "PK":
            task_file_path = os.path.join(root_task_path, f"{model_name}_parametric_knowledge_{data_version}.jsonl")
        elif args.task_type == "PCK":
            task_file_path = os.path.join(root_task_path, f"{model_name}_parametriccontextual_knowledge_{data_version}.jsonl")
        elif args.task_type == "RAG":
            task_file_path = os.path.join(root_task_path, f"{model_name}_rag_{data_version}.jsonl")
        else:
            raise Exception("Undefined task type: " + args.task_type + " or data version: " + data_version)
    dataset = load_dataset("json", data_files=task_file_path)["train"]

    if args.pilot_run:
        dataset = dataset.shuffle(seed=42).select(range(10))

    # run prediction
    if is_api_model:
        pred_res = generate_text_for_dataset_api(dataset, task=args.task_type, client=client, api_model_name=API_MODELS[model_name], max_length=100)
    elif "KF" not in args.task_type:
        pred_res = generate_text_for_dataset(dataset, task=args.task_type, generator=generator, max_length=100, eos="</answer>")
    else:
        pred_res = generate_text_for_dataset(dataset, task=args.task_type, generator=generator, max_length=100)

    if args.save_dir is None:
        if args.pilot_run:
            pred_res.to_json(os.path.join(os.environ["base_dir"], "output", "pilotruns", f"{model_name}_{args.task_type}_{data_version}.jsonl"))
        else:
            if args.mult_choice:
                pred_res.to_json(os.path.join(os.environ["base_dir"], "output", f"{model_name}_{args.task_type}_{data_version}_choice.jsonl"))
            else:
                pred_res.to_json(os.path.join(os.environ["base_dir"], "output", f"{model_name}_{args.task_type}_{data_version}_free.jsonl"))
    else:
        pred_res.to_json(args.save_dir)
