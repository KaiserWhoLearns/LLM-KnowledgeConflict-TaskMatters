import re
import os
import sys
import math
import pdb
from prettytable import PrettyTable
import pandas as pd
sys.path.append(os.getcwd())
import argparse
import statistics   
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset

load_dotenv()


def create_acc_row(test_model_name, task_type, data_version="", format="mult", target_metric=None):
    """
    Load the evaluation result of given model from /output/metrics, and generate a table of
     Performance (acc): NC HPCE HPC LPC Overall
    target_metric = f1, em, overallem, it would be effective only in the case of KF
    """
    if format != "mult" or task_type == "KFextract":
        if data_version == "":
            # final data/no version
            data_path = os.path.join(os.environ["base_dir"], "output", "metrics", f"{test_model_name}_{task_type}_free.jsonl")
        else:
            data_path = os.path.join(os.environ["base_dir"], "output", "metrics", f"{test_model_name}_{task_type}_{data_version}_free.jsonl")
        # Load the evaluation results
        try:
            dataset = load_dataset("json", data_files=data_path)["train"]
        except:
            print("No such data.", data_path)
            return {}
    else:
        if data_version == "":
            # final data/no version
            data_path = os.path.join(os.environ["base_dir"], "output", "metrics_mult", f"{test_model_name}_{task_type}_choice.jsonl")
        else:
            data_path = os.path.join(os.environ["base_dir"], "output", "metrics_mult", f"{test_model_name}_{task_type}_{data_version}_choice.jsonl")
        # Load the evaluation results
        try:
            dataset = load_dataset("json", data_files=data_path)["train"]
        except:
            print("No such data.", data_path)
            return {}

    # For each evidence type, compute metrics
    row = {"metric": target_metric}
    overall_metrics = []
    for evidence_type in ["NC", "HPC", "HPCE", "LPC"]:
        metrics = []
        for instance in dataset:
            if instance["context_type"] == evidence_type:
                if target_metric is None or target_metric == "f1":
                    metrics.append(instance["metrics"]["f1"])
                else:
                    metrics.append(instance["metrics"][target_metric])
        # Add to row
        row[evidence_type] = sum(metrics) / len(metrics) * 100
        row[f"{evidence_type}_std"] = (
            statistics.stdev(metrics) * 100 if len(metrics) > 1 else 0.0
        )
        row[f"{evidence_type}_sem"] = (
            statistics.stdev(metrics) * 100 / math.sqrt(len(metrics))  if len(metrics) > 1 else 0.0
        )
        overall_metrics += metrics
    row["Overall"] = sum(overall_metrics) / len(overall_metrics) * 100
    return row

def create_acc_table(test_model_name, format="mult", data_version=""):
    """
    Load the evaluation result of the given model from /output/metrics, and generate a table of
     Performance (acc): NC HPCE HPC LPC Overall
     """
    # For each task (KF, CK, PK), generate row
    tab =[]
    for task_type in ["KFextract", "CK", "PK", "PCK", "RAG"]:
        if format == "mult":
            target_metrics = ["f1", "exact_match"]
        else:
            if "KF" in task_type:
                target_metrics = ["f1", "exact_match", "strict_exact_match"]
            else:
                target_metrics = ["score", "f1"]
        for target_metric in target_metrics:
            row = create_acc_row(test_model_name, task_type, data_version=data_version, target_metric=target_metric, format=format)
            row["task"] = task_type
            tab.append(row)
    df = pd.DataFrame(tab)

    # save to CSV
    df.to_csv(os.path.join(os.environ['base_dir'], "results", f"{test_model_name}_{data_version}_{format}_perf.csv"), index=False)

    # Step 4: Display with PrettyTable
    table = PrettyTable()
    table.field_names = df.columns.tolist()

    for row in df.itertuples(index=False):
        table.add_row(row)
    print(table)

def create_len_ablation_row(test_model_name, task_type, data_version="", format="mult", target_metric=None, is_ablation=False):
    """
    Load the evaluation result for length ablation, handling both normal and ablation files
    """
    if format != "mult" or task_type == "KFextract":
        return {}  # Length ablation only for multiple choice format
    
    if is_ablation:
        # Load length ablation file
        data_path = os.path.join(os.environ["base_dir"], "output", "metrics_mult", 
                                f"{test_model_name}_{task_type}_{data_version}_choice_len_ablation.jsonl")
    else:
        # Load normal file
        data_path = os.path.join(os.environ["base_dir"], "output", "metrics_mult", 
                                f"{test_model_name}_{task_type}_{data_version}_choice.jsonl")
    
    try:
        dataset = load_dataset("json", data_files=data_path)["train"]
    except:
        print("No such data.", data_path)
        return {}

    # For each evidence type, compute metrics
    row = {"metric": target_metric}
    overall_metrics = []
    
    # For ablation files, only process HPC (which will be renamed to HPC-double)
    # For normal files, process HPC and HPCE
    if is_ablation:
        evidence_types = ["HPC"]
    else:
        evidence_types = ["HPC", "HPCE"]
    
    for evidence_type in evidence_types:
        metrics = []
        for instance in dataset:
            if instance["context_type"] == evidence_type:
                if target_metric is None or target_metric == "f1":
                    metrics.append(instance["metrics"]["f1"])
                else:
                    metrics.append(instance["metrics"][target_metric])
        
        # Rename HPC to HPC-double for ablation files
        col_name = "HPC-double" if (is_ablation and evidence_type == "HPC") else evidence_type
        
        if metrics:
            row[col_name] = sum(metrics) / len(metrics) * 100
            row[f"{col_name}_std"] = (
                statistics.stdev(metrics) * 100 if len(metrics) > 1 else 0.0
            )
            row[f"{col_name}_sem"] = (
                statistics.stdev(metrics) * 100 / math.sqrt(len(metrics)) if len(metrics) > 1 else 0.0
            )
            overall_metrics += metrics
    
    if overall_metrics:
        row["Overall"] = sum(overall_metrics) / len(overall_metrics) * 100
    
    return row

def create_len_ablation_table(test_model_name, format="mult", data_version=""):
    """
    Create table combining HPC (normal), HPC-double (from length ablation), and HPCE results
    """
    if format != "mult":
        print("Length ablation only supported for multiple choice format")
        return
    
    tab = []
    for task_type in ["CK", "PK", "PCK", "RAG"]:
        for target_metric in ["f1", "exact_match"]:
            # Get normal results (HPC and HPCE)
            normal_row = create_len_ablation_row(test_model_name, task_type, 
                                               data_version=data_version, 
                                               target_metric=target_metric, 
                                               format=format, 
                                               is_ablation=False)
            
            # Get ablation results (HPC-double)
            ablation_row = create_len_ablation_row(test_model_name, task_type, 
                                                  data_version=data_version, 
                                                  target_metric=target_metric, 
                                                  format=format, 
                                                  is_ablation=True)
            
            # Combine results
            if normal_row or ablation_row:
                combined_row = {"metric": target_metric, "task": task_type}
                
                # Add HPC from normal results
                if "HPC" in normal_row:
                    combined_row["HPC"] = normal_row["HPC"]
                    combined_row["HPC_std"] = normal_row["HPC_std"]
                    combined_row["HPC_sem"] = normal_row["HPC_sem"]
                
                # Add HPC-double from ablation results
                if "HPC-double" in ablation_row:
                    combined_row["HPC-double"] = ablation_row["HPC-double"]
                    combined_row["HPC-double_std"] = ablation_row["HPC-double_std"]
                    combined_row["HPC-double_sem"] = ablation_row["HPC-double_sem"]
                
                # Add HPCE from normal results
                if "HPCE" in normal_row:
                    combined_row["HPCE"] = normal_row["HPCE"]
                    combined_row["HPCE_std"] = normal_row["HPCE_std"]
                    combined_row["HPCE_sem"] = normal_row["HPCE_sem"]
                
                tab.append(combined_row)
    
    if tab:
        df = pd.DataFrame(tab)
        
        # Save to CSV
        output_path = os.path.join(os.environ['base_dir'], "results", 
                                 f"{test_model_name}_{data_version}_{format}_perf_len_ablation.csv")
        df.to_csv(output_path, index=False)
        print(f"Length ablation results saved to: {output_path}")
        
        # Display with PrettyTable
        table = PrettyTable()
        table.field_names = df.columns.tolist()
        
        for row in df.itertuples(index=False):
            table.add_row(row)
        print(table)

def create_prompt_ablation_row(test_model_name, task_type, data_version="", format="mult", target_metric=None, prompt_strength="neutral"):
    """
    Load the evaluation result for prompt ablation, handling both normal and ablation files with different strengths
    prompt_strength: None for normal files, or "weak", "neutral", "strong" for ablation files
    """
    if format != "mult" or task_type == "KFextract":
        return {}  # Prompt ablation only for multiple choice format
    
    # Load prompt ablation file with specific strength
    data_path = os.path.join(os.environ["base_dir"], "output", "metrics_mult",
                                f"{test_model_name}_{task_type}_{data_version}_choice_prompt_ablation_{prompt_strength}.jsonl")
    
    try:
        dataset = load_dataset("json", data_files=data_path)["train"]
    except:
        print("No such data.", data_path)
        return {}

    # For each evidence type, compute metrics
    row = {"metric": target_metric}
    overall_metrics = []
    
    # Process all evidence types for both normal and ablation files
    evidence_types = ["NC", "HPC", "HPCE", "LPC"]
    for evidence_type in evidence_types:
        metrics = []
        for instance in dataset:
            if instance["context_type"] == evidence_type:
                if target_metric is None or target_metric == "f1":
                    metrics.append(instance["metrics"]["f1"])
                else:
                    metrics.append(instance["metrics"][target_metric])
        
        # Add suffix for ablation results based on strength
        if prompt_strength:
            col_name = f"{evidence_type}-{prompt_strength}"
        else:
            col_name = evidence_type
        
        if metrics:
            row[col_name] = sum(metrics) / len(metrics) * 100
            row[f"{col_name}_std"] = (
                statistics.stdev(metrics) * 100 if len(metrics) > 1 else 0.0
            )
            row[f"{col_name}_sem"] = (
                statistics.stdev(metrics) * 100 / math.sqrt(len(metrics)) if len(metrics) > 1 else 0.0
            )
            overall_metrics += metrics
    
    if overall_metrics:
        if prompt_strength:
            row[f"Overall-{prompt_strength}"] = sum(overall_metrics) / len(overall_metrics) * 100
        else:
            row["Overall"] = sum(overall_metrics) / len(overall_metrics) * 100
    
    return row

def create_prompt_ablation_table(test_model_name, format="mult", data_version=""):
    """
    Create table comparing prompt ablation results for all strengths (weak, neutral, strong)
    """
    if format != "mult":
        print("Prompt ablation only supported for multiple choice format")
        return
    
    tab = []
    prompt_strengths = ["weak", "neutral", "strong"]
    
    for task_type in ["CK", "PK", "PCK", "RAG"]:
        for target_metric in ["f1", "exact_match"]:
            # Initialize combined row
            combined_row = {"metric": target_metric, "task": task_type}
            
            # Get ablation results for each strength
            for strength in prompt_strengths:
                ablation_row = create_prompt_ablation_row(test_model_name, task_type, 
                                                         data_version=data_version, 
                                                         target_metric=target_metric, 
                                                         format=format, 
                                                         prompt_strength=strength)
                
                # Add ablation results with strength suffix
                if ablation_row:
                    for evidence_type in ["NC", "HPC", "HPCE", "LPC"]:
                        col_name = f"{evidence_type}-{strength}"
                        if col_name in ablation_row:
                            combined_row[col_name] = ablation_row[col_name]
                            combined_row[f"{col_name}_std"] = ablation_row[f"{col_name}_std"]
                            combined_row[f"{col_name}_sem"] = ablation_row[f"{col_name}_sem"]
                    
                    # Add overall score for this strength
                    overall_col = f"Overall-{strength}"
                    if overall_col in ablation_row:
                        combined_row[overall_col] = ablation_row[overall_col]
            
            # Only add row if we have some data
            if len(combined_row) > 2:  # More than just metric and task
                tab.append(combined_row)
    
    if tab:
        df = pd.DataFrame(tab)
        
        # Save to CSV
        output_path = os.path.join(os.environ['base_dir'], "results", 
                                 f"{test_model_name}_{data_version}_{format}_perf_prompt_ablation.csv")
        df.to_csv(output_path, index=False)
        print(f"Prompt ablation results saved to: {output_path}")
        
        # Display with PrettyTable (selecting key columns for display)
        table = PrettyTable()
        
        # Select columns to display (task, metric, and ablation values)
        display_cols = ["task", "metric"]
        
        # Add evidence type columns for all strengths
        for evidence_type in ["NC", "HPC", "HPCE", "LPC"]:
            for strength in prompt_strengths:
                col_name = f"{evidence_type}-{strength}"
                if col_name in df.columns:
                    display_cols.append(col_name)
        
        # Add overall columns for all strengths
        for strength in prompt_strengths:
            overall_col = f"Overall-{strength}"
            if overall_col in df.columns:
                display_cols.append(overall_col)
        
        # Filter to only existing columns
        display_cols = [col for col in display_cols if col in df.columns]
        
        table.field_names = display_cols
        
        for _, row in df[display_cols].iterrows():
            table.add_row([row[col] for col in display_cols])
        print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="mistral7B",
                            help='name of a dataset')
    parser.add_argument('--data_version', type=str, default="full_v2", help='The version of the dataset to be generated.')
    parser.add_argument('--format', type=str, default="mult", help='multiple choice (mult) or free generation (free)')
    parser.add_argument('--len_ablation', action='store_true', help='Aggregate length ablation results')
    parser.add_argument('--prompt_ablation', action='store_true', help='Aggregate prompt ablation results')
    args = parser.parse_args()
    data_version = args.data_version
    
    # Check that len_ablation and prompt_ablation are not both true
    if args.len_ablation and args.prompt_ablation:
        parser.error("--len_ablation and --prompt_ablation cannot both be enabled")
    
    if args.len_ablation:
        create_len_ablation_table(args.test_model_name, format=args.format, data_version=data_version)
    elif args.prompt_ablation:
        create_prompt_ablation_table(args.test_model_name, format=args.format, data_version=data_version)
    else:
        create_acc_table(args.test_model_name, format=args.format, data_version=data_version)