import re
import os
import sys
import pdb
from prettytable import PrettyTable
import pandas as pd
sys.path.append(os.getcwd())
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset

load_dotenv()


def create_acc_row(test_model_name, task_type, data_version="", target_metric=None):
    """
    Load the evaluation result of given model from /output/metrics, and generate a table of
     Performance (acc): NC HPCE HPC LPC Overall
    target_metric = f1, em, overallem, it would be effective only in the case of KF
    """
    if data_version == "":
        # final data/no version
        data_path = os.path.join(os.environ["base_dir"], "output", "metrics", f"{test_model_name}_{task_type}.jsonl")
    else:
        data_path = os.path.join(os.environ["base_dir"], "output", "metrics", f"{test_model_name}_{task_type}_{data_version}.jsonl")
    # Load the evaluation results
    try:
        dataset = load_dataset("json", data_files=data_path)["train"]
    except:
        print("No such data.", data_path)
        return {}
    # For each evidence type, compute metrics
    row = {"metric": target_metric}
    overall_metrics = []
    metrics = []
    for evidence_type in ["NC", "HPC", "HPCE", "LPC"]:
        for instance in dataset:
            if instance["context_type"] == evidence_type:
                if target_metric is None or target_metric == "f1":
                    metrics.append(instance["metrics"]["f1"])
                else:
                    metrics.append(instance["metrics"][target_metric])
        # Add to row
        row[evidence_type] = sum(metrics) / len(metrics) * 100
        overall_metrics += metrics
    row["Overall"] = sum(overall_metrics) / len(overall_metrics) * 100
    return row

def create_acc_table(test_model_name, data_version=""):
    """
    Load the evaluation result of the given model from /output/metrics, and generate a table of
     Performance (acc): NC HPCE HPC LPC Overall
     """
    # For each task (KF, CK, PK), generate row
    tab =[]
    for task_type in ["KFextract", "CK", "PK", "PCK", "RAG"]:
        if "KF" in task_type:
            target_metrics = ["f1", "exact_match", "strict_exact_match"]
        else:
            target_metrics = ["score"]
        for target_metric in target_metrics:
            row = create_acc_row(test_model_name, task_type, data_version=data_version, target_metric=target_metric)
            row["task"] = task_type
            tab.append(row)
    df = pd.DataFrame(tab)

    # save to CSV
    df.to_csv(os.path.join(os.environ['base_dir'], "results", f"{test_model_name}_{data_version}_perf.csv"), index=False)

    # Step 4: Display with PrettyTable
    table = PrettyTable()
    table.field_names = df.columns.tolist()

    for row in df.itertuples(index=False):
        table.add_row(row)
    print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--data_version', type=str, default="full_v2", help='The version of the dataset to be generated.')
    args = parser.parse_args()
    data_version = args.data_version
    
    create_acc_table(args.test_model_name, data_version=data_version)