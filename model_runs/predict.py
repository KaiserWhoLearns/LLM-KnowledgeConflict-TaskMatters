
import os
import sys
import pdb
sys.path.append(os.getcwd())
import argparse
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PRETTY_TO_MODEL_NAME = {
    "llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen7B-instruct": "Qwen/Qwen2.5-7B-Instruct-1M",
    "deepseek-llama8b" : "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}

CONTEXT_TYPES = ["NC", "HPC", "HPCE", "LPC"]
def generate_text_for_dataset(dataset, task, generator, max_length=150, eos=None):
    """
    task = {KF, CK, PK}
    """
    # Generate text for the entire dataset
    generated_texts = []
    for entry in dataset:
        for context_type in CONTEXT_TYPES:
            # Input, Output, Prediction, Context type, task type
            input_text = entry[f"{context_type}_{task}_input"]
            output = generator(input_text, max_new_tokens=max_length, num_return_sequences=1)

            # Remove the input text from output
            pred = output[0]['generated_text'][len(input_text):].strip()
            # Cut to eos
            if not eos:
                pred = pred.split(eos)[0] + eos if eos in pred else pred
            generated_texts.append({
                "input": entry[f"{context_type}_{task}_input"],
                "output": entry[f"{context_type}_{task}_output"],
                "pred": pred,
                "context_type": context_type,
                "task_type": task
            })
    return Dataset.from_list(generated_texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--task_type', type=str, default="PK",
                            help='type of task. = PK, CK, PCK, KF, RAG')
    parser.add_argument('--pilot_run', action="store_true",
                            help='whether this is a pilot run. When set to true, we only make prediction for 10 insances.')
    parser.add_argument('--data_path', type=str, default=None,
                            help='Load data from. If none, will load from default document name.')
    parser.add_argument('--save_dir', type=str, default=None,
                            help='save pred to')
    parser.add_argument('--data_version', type=str, default=None, help='The version of the dataset to be generated.')
    
    args = parser.parse_args()
    data_version = args.data_version
    model_name = args.test_model_name
    model = AutoModelForCausalLM.from_pretrained(PRETTY_TO_MODEL_NAME[model_name])
    tokenizer = AutoTokenizer.from_pretrained(PRETTY_TO_MODEL_NAME[model_name])

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load the corresponding data
    if args.data_path is not None:
        task_file_path = args.data_path
    else:
        if "KF" in args.task_type:
            task_file_path = os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_{args.task_type}_{data_version}.jsonl")
        elif args.task_type == "CK":
            task_file_path = os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_contextual_knowledge_{data_version}.jsonl")
        elif args.task_type == "PK":
            task_file_path = os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_parametric_knowledge_{data_version}.jsonl")
        elif args.task_type == "PCK":
            task_file_path = os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_parametriccontextual_knowledge_{data_version}.jsonl")
        elif args.task_type == "RAG":
            task_file_path = os.path.join(os.environ["data_dir"], "task_data", f"{model_name}_rag_{data_version}.jsonl")
        else:
            raise Exception("Undefined task type: " + args.task_type + " or data version: " + data_version)
    dataset = load_dataset("json", data_files=task_file_path)["train"]

    if args.pilot_run:
        dataset = dataset.shuffle(seed=42).select(range(10))

    # run prediction
    if "KF" not in args.task_type:
        pred_res = generate_text_for_dataset(dataset, task=args.task_type, generator=generator, max_length=200, eos="</answer>")
    else:
        pred_res = generate_text_for_dataset(dataset, task=args.task_type, generator=generator, max_length=200)

    if args.save_dir is None:
        if args.pilot_run:
            pred_res.to_json(os.path.join(os.environ["base_dir"], "output", "pilotruns", f"{model_name}_{args.task_type}_{data_version}.jsonl"))
        else:
            pred_res.to_json(os.path.join(os.environ["base_dir"], "output", f"{model_name}_{args.task_type}_{data_version}.jsonl"))
    else:
        pred_res.to_json(args.save_dir)
