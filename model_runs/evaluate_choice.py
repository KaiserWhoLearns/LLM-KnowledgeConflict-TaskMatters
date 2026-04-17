import re
import os
import sys
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

def f1_score_per_instance(pred, gold):
    pred_set = set(pred)
    gold_set = set(gold)
    true_positives = len(pred_set & gold_set)
    
    if not pred_set and not gold_set:
        return 1.0  # both empty, considered perfect match
    
    if true_positives == 0:
        return 0.0
    
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def average_f1(predictions, golds):
    f1s = [f1_score_per_instance(p, g) for p, g in zip(predictions, golds)]
    return sum(f1s) / len(f1s)

def eval_PKCK(prediction, answer):
    """
    prediction: a list of options ([A, B])
    answer: a list of options ([A, B])
    """
    return {"f1": f1_score_per_instance(prediction, answer), "exact_match": int(set(prediction) == set(answer))}

def extract_choices(text):
    # Match multiple patterns:
    # 1. Parentheses: (A), ( AB), ( AC )
    # 2. Letter with period: A., B., C., D.
    # 3. Letter with period and word: A.word, B.something, D.Fishing
    
    results = []
    
    # Pattern 1: Match patterns like (A), ( AB), ( AC )
    paren_pattern = r'\(\s*([A-D](?:\s*[A-D])*)\s*\)'
    paren_matches = re.findall(paren_pattern, text)
    
    for match in paren_matches:
        # Split by whitespace to handle things like "A B" or "AC"
        letters = re.findall(r'[A-D]', match)
        if letters:
            results += letters
    
    # Pattern 2 & 3: Match A., B., C., D. or A.word, B.something, D.Fishing
    # Look for capital letters A-D followed by a period (and optionally a word)
    period_pattern = r'\b([A-D])\.(?:\w+)?'
    period_matches = re.findall(period_pattern, text)
    results += period_matches
    
    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique_results.append(item)
    
    return unique_results


def evaluate_full(orig_path, dataset):
    metrics = dict()
    metrics = []
    questions = []
    extract_question_text = lambda text: text.rsplit("Question: ", 1)[-1].split("\nContext", 1)[0].strip() if "Question: " in text and "\nAnswer: " in text else None
    cleaned_pred = []

    for instance in dataset:
        question = extract_question_text(instance["input"])
        questions.append(question)

        # Parse for prediction
        pred = extract_choices(instance["pred"])
        # pdb.set_trace()
        cleaned_pred.append(pred)
        # Convert answers into list
        answers = list(instance["output"])

        if question is None:
            print("Error: No question is contained in this example. Input = ", instance["input"])
        metrics.append(eval_PKCK(prediction=pred, answer=answers))

    print(metrics)
    dataset = dataset.add_column("metrics", metrics)
    dataset = dataset.add_column("cleaned_pred", cleaned_pred)
    dataset = dataset.add_column("question", questions)
    # Save
    dataset.to_json(os.path.join(os.environ["base_dir"], "output", "metrics_mult", orig_path.split("/")[-1].split(".json")[0] + ".jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="mistral7B",
                            help='name of a dataset')
    parser.add_argument('--task_type', type=str, default="RAG",
                            help='type of task. = PK, CK, KF, RAG')
    parser.add_argument('--pred_path', type=str, default=None,
                            help='load prediction from')
    args = parser.parse_args()
    model_name = args.test_model_name
    
    # Load the predictions
    dataset = load_dataset("json", data_files=args.pred_path)["train"]
    # # # Sample for 10 instances
    # dataset = dataset.shuffle(seed=42).select(range(100))

    # Evalaute
    evaluate_full(orig_path=args.pred_path, dataset=dataset)