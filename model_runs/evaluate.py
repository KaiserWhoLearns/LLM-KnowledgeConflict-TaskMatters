import re
import os
import sys
import pdb
from openai import OpenAI
from together import Together
sys.path.append(os.getcwd())
import argparse
import string
from collections import Counter
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset

load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
together_client = Together(api_key=TOGETHER_API_KEY)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def eval_kf_extraction(prediction, acceptable_answers):
    """
    # F1: for each prediction, take highest F1 with the set of possible answers
    # EM: as long as one of the extracted answer is in the set of acceptable answers, EM=1
    # FullEM: The model is able to extract all acceptable answer
    """
    # Strip for predicted answers
    raw_preds = prediction.split(".")
    preds = [normalize_answer(raw_pred) for raw_pred in raw_preds]
    f1s = []
    em = 0
    fullem = 1 if len(preds) > 0 else 0
    for pred in preds:
        for ans in acceptable_answers:
            # Compute F1
            f1s.append(f1_score(prediction=pred, ground_truth=ans))
        if pred in acceptable_answers:
            em = 1
        if pred not in acceptable_answers:
            fullem = 0
    return {"f1": max(f1s), "exact_match": em, "strict_exact_match": fullem}

# def eval_CK(question, prediction, answer, eval_model="openai"):
#     # Load the prompt from txt file
#     with open(os.path.join(os.environ["base_dir"], "prompts", "eval_ck.txt"), 'r', encoding='utf-8') as file:
#         prompt = file.read()
#     content = f"###Question: {question}\n###Response: {prediction}\n###Answer: {answer}"
#     # Send OpenAI request
#     if eval_model == "openai":
#         completion = openai_client.chat.completions.create(
#                 model="gpt-4-1106-preview",
#                 messages=[
#                     {"role": "developer", "content": prompt},
#                     {
#                         "role": "user",
#                         "content": content
#                     }
#                 ]
#         )
#         response = completion.choices[0].message.content
#     else:
#         response = together_client.chat.completions.create(
#             model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
#             messages=[{"role": "user", "content": prompt + content}],
#         ).choices[0].message.content
#         try:
#             response = response.split("</think>")[1]
#         except:
#             # Unjudgable instance, model does not think
#             return False
#     # TODO: Save full response
#     return {"score": 0, "response": response} if "incorrect" in response.lower() else {"score": 1, "response": response}

def eval_PK(question, prediction, answer, eval_model="openai"):
    # PK CK share the same evaluator
    # Load the prompt from txt file
    with open(os.path.join(os.environ["base_dir"], "prompts", "eval_pk.txt"), 'r', encoding='utf-8') as file:
        prompt = file.read()

    content = f"###Question: {question}\n###Response: {prediction}\n###Answer: {answer}\n"
    # Send OpenAI request
    if eval_model == "openai":
        completion = openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "developer", "content": prompt},
                    {
                        "role": "user",
                        "content": content
                    }
                ]
        )
        response = completion.choices[0].message.content
    else:
        response = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[{"role": "user", "content": prompt + content}],
        ).choices[0].message.content
        try:
            response = response.split("</think>")[1]
        except:
            # Unjudgable instance, model does not think
            return False
    # pdb.set_trace()
    if "incorrect" in response.lower():
        return {"score": 0, "response": response}
    return {"score": 1, "response": response}

def eval_RAGPCK(question, prediction, answer, eval_model="openai", task_type="PCK"):
    # RAG PCK Share the same evaluator
    # Load the prompt from txt file
    # TODO: Rewrite the input format
    if task_type == "PCK":
        with open(os.path.join(os.environ["base_dir"], "prompts", "eval_pck.txt"), 'r', encoding='utf-8') as file:
            prompt = file.read()
    else:
        with open(os.path.join(os.environ["base_dir"], "prompts", "eval_rag.txt"), 'r', encoding='utf-8') as file:
            prompt = file.read()
    # Send OpenAI request
    content = f"###Question: {question}\n###Response: {prediction}\n###Answer: {answer} + \n###Comment: "
    if eval_model == "openai":
        completion = openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "developer", "content": prompt},
                    {
                        "role": "user",
                        "content": content
                    }
                ]
        )
        response = completion.choices[0].message.content
    else:
        response = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            messages=[{"role": "user", "content": prompt + content}],
        ).choices[0].message.content
        try:
            response = response.split("</think>")[1]
        except:
            # Unjudgable instance, model does not think
            return False
    # pdb.set_trace()
    if "incorrect" in response.lower():
        return {"score": 0, "response": response}
    elif "partially correct" in response.lower():
        return {"score": 0.5, "response": response}
    return {"score": 1, "response": response}

def evaluate_full(orig_path, dataset):
    metrics = dict()
    metrics = []
    questions = []
    extract_question_text = lambda text: text.rsplit("Question: ", 1)[-1].split("\nContext", 1)[0].strip() if "Question: " in text and "\nAnswer: " in text else None

    for instance in dataset:
        question = extract_question_text(instance["input"])
        questions.append(question)
        if question is None:
            print("Error: No question is contained in this example. Input = ", instance["input"])
        # pdb.set_trace()
        if instance["task_type"] == "KFextract":
            metrics.append(eval_kf_extraction(prediction=instance["pred"], acceptable_answers=instance["output"]))
        elif instance["task_type"] == "PK" or instance["task_type"] == "CK":
            metrics.append(eval_PK(question=question, prediction=instance["pred"], answer=instance["output"], eval_model="openai"))
        # elif instance["task_type"] == "CK":
        #     metrics.append(eval_CK(question=question, prediction=instance["pred"], answer=instance["output"], eval_model="openai"))
        elif instance["task_type"] == "PCK" or instance["task_type"] == "RAG":
            metrics.append(eval_RAGPCK(question=question, prediction=instance["pred"], answer=instance["output"], eval_model="openai", task_type=instance["task_type"]))
        else:
            raise Exception(f"The given task type ({instance['task_type']}) is not supported.")
    dataset = dataset.add_column("metrics", metrics)
    dataset = dataset.add_column("question", questions)
    # Save
    if "pilot" not in orig_path:
        dataset.to_json(os.path.join(os.environ["base_dir"], "output", "metrics_wq", orig_path.split("/")[-1].split(".json")[0] + ".jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--task_type', type=str, default="PK",
                            help='type of task. = PK, CK, KF, RAG')
    parser.add_argument('--pred_path', type=str, default=None,
                            help='load prediction from')
    args = parser.parse_args()
    model_name = args.test_model_name
    
    # Load the predictions
    dataset = load_dataset("json", data_files=args.pred_path)["train"]
    # Sample for 10 instances
    dataset = dataset.shuffle(seed=42).select(range(10))

    # Evalaute
    evaluate_full(orig_path=args.pred_path, dataset=dataset)