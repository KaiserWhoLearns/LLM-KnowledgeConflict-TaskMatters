import json
import os
from typing import Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Match styling in analysis/make_plots.py so fonts agree across figures.
font = {'family': 'serif', 'size': 19}
mpl.rc('font', **font)
plt.rcParams["font.family"] = "Nimbus Roman"


def count_error_types(model_name: str, base_dir: str = "/scratch4/mdredze1/hsun74/KnowledgeInstruct/output", include_len_ablation: bool = True, task: str = "RAG") -> Dict[str, int]:
    """
    Count error types for RAG/PCK tasks by comparing predictions with correct answers.
    
    For each context type (NC, HPC, HPCE, HPC double), we track:
    1. Get NC part correct but HPC/HPCE part wrong
    2. Get HPC/HPCE part correct but NC part wrong
    3. Get both parts wrong
    4. Get both parts correct
    
    Args:
        model_name: Name of the model (e.g., "mistral7B", "olmo2-7B")
        base_dir: Base directory containing output files
        include_len_ablation: Whether to include HPC double from len_ablation files
        task: Task type, either "RAG" or "PCK"
        
    Returns:
        Dictionary with error counts by type
    """
    # Initialize counters for each context type
    stats = {
        # Totals
        "total_nc": 0,
        "total_hpc": 0,
        "total_hpce": 0,
        "total_lpc": 0,
        "total_hpc_double": 0,
        # NC context (only has NC answer)
        "nc_correct": 0,
        "nc_wrong": 0,
        # HPC context
        "hpc_both_correct": 0,
        "hpc_nc_correct_hpc_wrong": 0,
        "hpc_hpc_correct_nc_wrong": 0,
        "hpc_both_wrong": 0,
        # HPCE context
        "hpce_both_correct": 0,
        "hpce_nc_correct_hpce_wrong": 0,
        "hpce_hpce_correct_nc_wrong": 0,
        "hpce_both_wrong": 0,
        # HPC double context
        "hpc_double_both_correct": 0,
        "hpc_double_nc_correct_hpc_wrong": 0,
        "hpc_double_hpc_correct_nc_wrong": 0,
        "hpc_double_both_wrong": 0,
    }
    
    # Read predictions from metrics_mult
    predictions_file = os.path.join(base_dir, "metrics_mult", f"{model_name}_{task}_full_v2_choice.jsonl")
    
    # Read ground truth from model task file
    ground_truth_file = os.path.join(base_dir, f"{model_name}_{task}_full_v2_choice.jsonl")
    
    # Load ground truth data
    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'context_type' in data and 'output' in data:
                # Create unique key based on question and context type
                key = (data.get('input', '').split('Question: ')[-1].split('\n')[0], data['context_type'])
                ground_truth[key] = set(data['output'])
    
    # Process predictions
    with open(predictions_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            context_type = data.get('context_type', '')
            question = data.get('question', '')
            cleaned_pred = set(data.get('cleaned_pred', []))
            
            # Count totals by context type
            if context_type == 'NC':
                stats['total_nc'] += 1
                # For NC context, there's only NC answer
                nc_ground_truth = ground_truth.get((question, 'NC'), set())
                if len(cleaned_pred & nc_ground_truth) > 0:
                    stats['nc_correct'] += 1
                else:
                    stats['nc_wrong'] += 1
                    
            elif context_type == 'HPC':
                stats['total_hpc'] += 1
                # Get ground truths
                hpc_ground_truth = ground_truth.get((question, 'HPC'), set())
                nc_ground_truth = ground_truth.get((question, 'NC'), set())
                
                # Determine which parts of HPC ground truth correspond to NC vs HPC
                nc_part = hpc_ground_truth & nc_ground_truth
                hpc_part = hpc_ground_truth - nc_part
                
                # Check what the model got right
                got_nc = len(cleaned_pred & nc_part) > 0
                got_hpc = len(cleaned_pred & hpc_part) > 0
                
                # Categorize
                if got_nc and got_hpc:
                    stats['hpc_both_correct'] += 1
                elif got_nc and not got_hpc:
                    stats['hpc_nc_correct_hpc_wrong'] += 1
                elif not got_nc and got_hpc:
                    stats['hpc_hpc_correct_nc_wrong'] += 1
                else:
                    stats['hpc_both_wrong'] += 1
                    
            elif context_type == 'HPCE':
                stats['total_hpce'] += 1
                # Get ground truths
                hpce_ground_truth = ground_truth.get((question, 'HPCE'), set())
                nc_ground_truth = ground_truth.get((question, 'NC'), set())
                
                # Determine which parts of HPCE ground truth correspond to NC vs HPCE
                nc_part = hpce_ground_truth & nc_ground_truth
                hpce_part = hpce_ground_truth - nc_part
                
                # Check what the model got right
                got_nc = len(cleaned_pred & nc_part) > 0
                got_hpce = len(cleaned_pred & hpce_part) > 0
                
                # Categorize
                if got_nc and got_hpce:
                    stats['hpce_both_correct'] += 1
                elif got_nc and not got_hpce:
                    stats['hpce_nc_correct_hpce_wrong'] += 1
                elif not got_nc and got_hpce:
                    stats['hpce_hpce_correct_nc_wrong'] += 1
                else:
                    stats['hpce_both_wrong'] += 1
                    
            elif context_type == 'LPC':
                stats['total_lpc'] += 1
    
    # Process len_ablation file for HPC double
    if include_len_ablation:
        len_ablation_predictions_file = os.path.join(base_dir, "metrics_mult", f"{model_name}_{task}_full_v2_choice_len_ablation.jsonl")
        len_ablation_ground_truth_file = os.path.join(base_dir, f"{model_name}_{task}_full_v2_choice_len_ablation.jsonl")
        
        # Check if len_ablation files exist
        if os.path.exists(len_ablation_predictions_file) and os.path.exists(len_ablation_ground_truth_file):
            # Load len_ablation ground truth
            len_ablation_ground_truth = {}
            with open(len_ablation_ground_truth_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'context_type' in data and 'output' in data:
                        key = (data.get('input', '').split('Question: ')[-1].split('\n')[0], data['context_type'])
                        len_ablation_ground_truth[key] = set(data['output'])
            
            # Process len_ablation predictions (only HPC instances)
            with open(len_ablation_predictions_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    context_type = data.get('context_type', '')
                    if context_type == 'HPC':  # Only process HPC from len_ablation
                        stats['total_hpc_double'] += 1
                        question = data.get('question', '')
                        cleaned_pred = set(data.get('cleaned_pred', []))
                        
                        # Get ground truths
                        hpc_ground_truth = len_ablation_ground_truth.get((question, 'HPC'), set())
                        # Use NC ground truth from original file
                        nc_ground_truth = ground_truth.get((question, 'NC'), set())
                        
                        # Determine which parts correspond to NC vs HPC
                        nc_part = hpc_ground_truth & nc_ground_truth
                        hpc_part = hpc_ground_truth - nc_part
                        
                        # Check what the model got right
                        got_nc = len(cleaned_pred & nc_part) > 0
                        got_hpc = len(cleaned_pred & hpc_part) > 0
                        
                        # Categorize
                        if got_nc and got_hpc:
                            stats['hpc_double_both_correct'] += 1
                        elif got_nc and not got_hpc:
                            stats['hpc_double_nc_correct_hpc_wrong'] += 1
                        elif not got_nc and got_hpc:
                            stats['hpc_double_hpc_correct_nc_wrong'] += 1
                        else:
                            stats['hpc_double_both_wrong'] += 1
    
    return stats


def analyze_all_models(base_dir: str = "/scratch4/mdredze1/hsun74/KnowledgeInstruct/output", task: str = "RAG") -> Dict[str, Dict[str, int]]:
    """
    Analyze error types for all models with RAG/PCK results.
    
    Args:
        base_dir: Base directory containing output files
        task: Task type, either "RAG" or "PCK"
        
    Returns:
        Dictionary mapping model names to their error statistics
    """
    # Find all models with task results
    metrics_dir = os.path.join(base_dir, "metrics_mult")
    models = set()
    
    for filename in os.listdir(metrics_dir):
        if filename.endswith(f"_{task}_full_v2_choice.jsonl") and not filename.endswith("_len_ablation.jsonl"):
            model_name = filename.replace(f"_{task}_full_v2_choice.jsonl", "")
            models.add(model_name)
    
    # Analyze each model
    results = {}
    for model in sorted(models):
        try:
            results[model] = count_error_types(model, base_dir, task=task)
        except Exception as e:
            print(f"Error processing {model}: {e}")
    
    return results


def print_error_analysis(results: Dict[str, Dict[str, int]], task: str = "RAG"):
    """
    Print formatted error analysis results.
    
    Args:
        results: Dictionary mapping model names to error statistics
        task: Task type, either "RAG" or "PCK"
    """
    print(f"\n{task} Task Error Analysis")
    print("="*80)
    
    for model, stats in results.items():
        print(f"\nModel: {model}")
        print("-"*40)
        print(f"Total samples - NC: {stats['total_nc']}, HPC: {stats['total_hpc']}, HPCE: {stats['total_hpce']}, LPC: {stats['total_lpc']}")
        if 'total_hpc_double' in stats:
            print(f"               HPC double: {stats['total_hpc_double']}")
        
        # NC context analysis
        if stats['total_nc'] > 0:
            print("\nNC Context Analysis:")
            print(f"  Correct: {stats['nc_correct']} ({stats['nc_correct']/stats['total_nc']*100:.1f}%)")
            print(f"  Wrong: {stats['nc_wrong']} ({stats['nc_wrong']/stats['total_nc']*100:.1f}%)")
        
        # HPC context analysis
        if stats['total_hpc'] > 0:
            print("\nHPC Context Analysis:")
            print(f"  Both correct: {stats['hpc_both_correct']} ({stats['hpc_both_correct']/stats['total_hpc']*100:.1f}%)")
            print(f"  NC correct, HPC wrong: {stats['hpc_nc_correct_hpc_wrong']} ({stats['hpc_nc_correct_hpc_wrong']/stats['total_hpc']*100:.1f}%)")
            print(f"  HPC correct, NC wrong: {stats['hpc_hpc_correct_nc_wrong']} ({stats['hpc_hpc_correct_nc_wrong']/stats['total_hpc']*100:.1f}%)")
            print(f"  Both wrong: {stats['hpc_both_wrong']} ({stats['hpc_both_wrong']/stats['total_hpc']*100:.1f}%)")
        
        # HPCE context analysis
        if stats['total_hpce'] > 0:
            print("\nHPCE Context Analysis:")
            print(f"  Both correct: {stats['hpce_both_correct']} ({stats['hpce_both_correct']/stats['total_hpce']*100:.1f}%)")
            print(f"  NC correct, HPCE wrong: {stats['hpce_nc_correct_hpce_wrong']} ({stats['hpce_nc_correct_hpce_wrong']/stats['total_hpce']*100:.1f}%)")
            print(f"  HPCE correct, NC wrong: {stats['hpce_hpce_correct_nc_wrong']} ({stats['hpce_hpce_correct_nc_wrong']/stats['total_hpce']*100:.1f}%)")
            print(f"  Both wrong: {stats['hpce_both_wrong']} ({stats['hpce_both_wrong']/stats['total_hpce']*100:.1f}%)")
        
        # HPC double context analysis
        if 'total_hpc_double' in stats and stats['total_hpc_double'] > 0:
            print("\nHPC double Context Analysis:")
            print(f"  Both correct: {stats['hpc_double_both_correct']} ({stats['hpc_double_both_correct']/stats['total_hpc_double']*100:.1f}%)")
            print(f"  NC correct, HPC wrong: {stats['hpc_double_nc_correct_hpc_wrong']} ({stats['hpc_double_nc_correct_hpc_wrong']/stats['total_hpc_double']*100:.1f}%)")
            print(f"  HPC correct, NC wrong: {stats['hpc_double_hpc_correct_nc_wrong']} ({stats['hpc_double_hpc_correct_nc_wrong']/stats['total_hpc_double']*100:.1f}%)")
            print(f"  Both wrong: {stats['hpc_double_both_wrong']} ({stats['hpc_double_both_wrong']/stats['total_hpc_double']*100:.1f}%)")
        
        # Summary across HPC+HPCE
        total_hpc_hpce = stats['total_hpc'] + stats['total_hpce']
        if total_hpc_hpce > 0:
            both_correct = stats['hpc_both_correct'] + stats['hpce_both_correct']
            nc_correct_other_wrong = stats['hpc_nc_correct_hpc_wrong'] + stats['hpce_nc_correct_hpce_wrong']
            other_correct_nc_wrong = stats['hpc_hpc_correct_nc_wrong'] + stats['hpce_hpce_correct_nc_wrong']
            both_wrong = stats['hpc_both_wrong'] + stats['hpce_both_wrong']
            
            print("\nSummary (HPC+HPCE combined):")
            print(f"  Both correct: {both_correct} ({both_correct/total_hpc_hpce*100:.1f}%)")
            print(f"  NC correct, HPC/HPCE wrong: {nc_correct_other_wrong} ({nc_correct_other_wrong/total_hpc_hpce*100:.1f}%)")
            print(f"  HPC/HPCE correct, NC wrong: {other_correct_nc_wrong} ({other_correct_nc_wrong/total_hpc_hpce*100:.1f}%)")
            print(f"  Both wrong: {both_wrong} ({both_wrong/total_hpc_hpce*100:.1f}%)")


def print_error_table(results: Dict[str, Dict[str, int]]):
    """
    Print error analysis as a table with columns [NC only, HPC only, Neither]
    and rows for each model's HPC, HPCE, and HPC double contexts.
    
    Args:
        results: Dictionary mapping model names to error statistics
    """
    print("\nError Analysis Table")
    print("="*90)
    print(f"{'Model + Context':<35} {'NC only':<15} {'HPC/HPCE only':<15} {'Neither':<15}")
    print("-"*90)
    
    for model, stats in results.items():
        # HPC row
        if stats['total_hpc'] > 0:
            nc_only = stats['hpc_nc_correct_hpc_wrong']
            hpc_only = stats['hpc_hpc_correct_nc_wrong']
            neither = stats['hpc_both_wrong']
            print(f"{model + ' (HPC)':<35} {nc_only:<15} {hpc_only:<15} {neither:<15}")
        
        # HPCE row
        if stats['total_hpce'] > 0:
            nc_only = stats['hpce_nc_correct_hpce_wrong']
            hpce_only = stats['hpce_hpce_correct_nc_wrong']
            neither = stats['hpce_both_wrong']
            print(f"{model + ' (HPCE)':<35} {nc_only:<15} {hpce_only:<15} {neither:<15}")
        
        # HPC double row
        if 'total_hpc_double' in stats and stats['total_hpc_double'] > 0:
            nc_only = stats['hpc_double_nc_correct_hpc_wrong']
            hpc_only = stats['hpc_double_hpc_correct_nc_wrong']
            neither = stats['hpc_double_both_wrong']
            print(f"{model + ' (HPC double)':<35} {nc_only:<15} {hpc_only:<15} {neither:<15}")
        
        print()  # Empty line between models


def visualize_error_analysis(results: Dict[str, Dict[str, int]], output_dir: str = "results/figures", task: str = "RAG"):
    """
    Create a stacked percentage bar plot for error analysis showing only NC Only and HPC/HPCE Only.
    Now includes HPC, HPCE, and HPC double bars.
    
    Args:
        results: Dictionary mapping model names to error statistics
        output_dir: Directory to save the figure
        task: Task type, either "RAG" or "PCK"
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    models = sorted(results.keys())
    n_models = len(models)
    
    # Map model names to pretty names (from plot_hpc_hpce_comparison.py)
    pretty_model_names = {
        'olmo2-7B': 'OLMo2-7B',
        'mistral7B': 'Mistral7B',
        'qwen7B-instruct': 'Qwen-7B',
        'olmo2-13B': 'OLMo2-13B',
        'qwen2.5-14B-instruct': 'Qwen-14B',
        'gpt5.2': 'GPT-5.2'
    }
    
    # Get pretty names for display
    display_names = [pretty_model_names.get(model, model) for model in models]
    
    # Data for HPC, HPCE, and HPC double bars
    hpc_nc_only = []
    hpc_only = []
    
    hpce_nc_only = []
    hpce_only = []
    
    hpc_double_nc_only = []
    hpc_double_only = []
    
    for model in models:
        stats = results[model]
        
        # HPC percentages (normalized to NC Only + HPC Only)
        total_hpc = stats['total_hpc']
        if total_hpc > 0:
            nc_only_count = stats['hpc_nc_correct_hpc_wrong']
            hpc_only_count = stats['hpc_hpc_correct_nc_wrong']
            total_shown = nc_only_count + hpc_only_count
            if total_shown > 0:
                hpc_nc_only.append(nc_only_count / total_shown * 100)
                hpc_only.append(hpc_only_count / total_shown * 100)
            else:
                hpc_nc_only.append(0)
                hpc_only.append(0)
        else:
            hpc_nc_only.append(0)
            hpc_only.append(0)
            
        # HPCE percentages (normalized to NC Only + HPCE Only)
        total_hpce = stats['total_hpce']
        if total_hpce > 0:
            nc_only_count = stats['hpce_nc_correct_hpce_wrong']
            hpce_only_count = stats['hpce_hpce_correct_nc_wrong']
            total_shown = nc_only_count + hpce_only_count
            if total_shown > 0:
                hpce_nc_only.append(nc_only_count / total_shown * 100)
                hpce_only.append(hpce_only_count / total_shown * 100)
            else:
                hpce_nc_only.append(0)
                hpce_only.append(0)
        else:
            hpce_nc_only.append(0)
            hpce_only.append(0)
            
        # HPC double percentages (normalized to NC Only + HPC Only)
        total_hpc_double = stats.get('total_hpc_double', 0)
        if total_hpc_double > 0:
            nc_only_count = stats['hpc_double_nc_correct_hpc_wrong']
            hpc_only_count = stats['hpc_double_hpc_correct_nc_wrong']
            total_shown = nc_only_count + hpc_only_count
            if total_shown > 0:
                hpc_double_nc_only.append(nc_only_count / total_shown * 100)
                hpc_double_only.append(hpc_only_count / total_shown * 100)
            else:
                hpc_double_nc_only.append(0)
                hpc_double_only.append(0)
        else:
            hpc_double_nc_only.append(0)
            hpc_double_only.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(n_models)
    width = 0.25
    
    # Colorblind-friendly colors (using Tol colorscheme)
    nc_color = '#4477AA'  # Blue for NC Only
    hpc_color = '#CC6677'  # Pink/Rose for HPC Only
    
    # Create stacked bars for HPC
    p1_hpc = ax.bar(x - width, hpc_nc_only, width, label='NC Only', color=nc_color)
    p2_hpc = ax.bar(x - width, hpc_only, width, bottom=hpc_nc_only, 
                    label='PC Only', color=hpc_color)
    
    # Create stacked bars for HPC double (middle position)
    p1_hpc_double = ax.bar(x, hpc_double_nc_only, width, color=nc_color)
    p2_hpc_double = ax.bar(x, hpc_double_only, width, bottom=hpc_double_nc_only, 
                          color=hpc_color)
    
    # Create stacked bars for HPCE
    p1_hpce = ax.bar(x + width, hpce_nc_only, width, color=nc_color)
    p2_hpce = ax.bar(x + width, hpce_only, width, bottom=hpce_nc_only, 
                     color=hpc_color)
    
    # Customize plot
    ax.set_ylabel('Percentage (%)', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Remove default x-axis labels
    ax.legend(loc='upper right', fontsize=18)
    ax.set_ylim(0, 100)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels to each bar portion
    for i in range(n_models):
        # HPC bar percentages
        if hpc_nc_only[i] > 5:  # Only show if large enough
            ax.text(i - width, hpc_nc_only[i]/2, f'{hpc_nc_only[i]:.0f}%', 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
        if hpc_only[i] > 5:
            ax.text(i - width, hpc_nc_only[i] + hpc_only[i]/2, f'{hpc_only[i]:.0f}%', 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
        
        # HPC double bar percentages
        if hpc_double_nc_only[i] > 5:
            ax.text(i, hpc_double_nc_only[i]/2, f'{hpc_double_nc_only[i]:.0f}%', 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
        if hpc_double_only[i] > 5:
            ax.text(i, hpc_double_nc_only[i] + hpc_double_only[i]/2, f'{hpc_double_only[i]:.0f}%', 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
        
        # HPCE bar percentages
        if hpce_nc_only[i] > 5:
            ax.text(i + width, hpce_nc_only[i]/2, f'{hpce_nc_only[i]:.0f}%', 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
        if hpce_only[i] > 5:
            ax.text(i + width, hpce_nc_only[i] + hpce_only[i]/2, f'{hpce_only[i]:.0f}%', 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
    
    # Add text labels for context types and model names
    for i in range(n_models):
        # Context labels (higher)
        ax.text(i - width, -3, 'HPC', ha='center', va='top', fontsize=12, color='gray')
        ax.text(i, -3, 'HPCdub', ha='center', va='top', fontsize=12, color='gray')
        ax.text(i + width, -3, 'HPCE', ha='center', va='top', fontsize=12, color='gray')
        # Model names (lower)
        ax.text(i, -8, display_names[i], ha='center', va='top', fontsize=18, color='black')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF
    output_path_pdf = os.path.join(output_dir, f'{task.lower()}_error_analysis.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Figure also saved as PDF: {output_path_pdf}")
    
    plt.close()


def create_pie_charts(results: Dict[str, Dict[str, int]], output_dir: str = "results/figures", task: str = "RAG"):
    """
    Create three pie charts showing the percentage of NC Only, PC Only, and both wrong instances
    averaged across models for HPC, HPCdub, and HPCE.
    
    Args:
        results: Dictionary mapping model names to error statistics
        output_dir: Directory to save the figure
        task: Task type, either "RAG" or "PCK"
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize accumulators for each context type
    hpc_totals = {'nc_only': 0, 'pc_only': 0, 'both_wrong': 0}
    hpcdub_totals = {'nc_only': 0, 'pc_only': 0, 'both_wrong': 0}
    hpce_totals = {'nc_only': 0, 'pc_only': 0, 'both_wrong': 0}
    
    n_models = len(results)
    
    # Aggregate data across all models
    for model, stats in results.items():
        # HPC data
        if stats['total_hpc'] > 0:
            # Calculate percentages excluding both_correct
            nc_only = stats['hpc_nc_correct_hpc_wrong']
            pc_only = stats['hpc_hpc_correct_nc_wrong']
            both_wrong = stats['hpc_both_wrong']
            total_errors = nc_only + pc_only + both_wrong
            
            if total_errors > 0:
                hpc_totals['nc_only'] += (nc_only / total_errors)
                hpc_totals['pc_only'] += (pc_only / total_errors)
                hpc_totals['both_wrong'] += (both_wrong / total_errors)
        
        # HPC double data
        if stats.get('total_hpc_double', 0) > 0:
            nc_only = stats['hpc_double_nc_correct_hpc_wrong']
            pc_only = stats['hpc_double_hpc_correct_nc_wrong']
            both_wrong = stats['hpc_double_both_wrong']
            total_errors = nc_only + pc_only + both_wrong
            
            if total_errors > 0:
                hpcdub_totals['nc_only'] += (nc_only / total_errors)
                hpcdub_totals['pc_only'] += (pc_only / total_errors)
                hpcdub_totals['both_wrong'] += (both_wrong / total_errors)
        
        # HPCE data
        if stats['total_hpce'] > 0:
            nc_only = stats['hpce_nc_correct_hpce_wrong']
            pc_only = stats['hpce_hpce_correct_nc_wrong']
            both_wrong = stats['hpce_both_wrong']
            total_errors = nc_only + pc_only + both_wrong
            
            if total_errors > 0:
                hpce_totals['nc_only'] += (nc_only / total_errors)
                hpce_totals['pc_only'] += (pc_only / total_errors)
                hpce_totals['both_wrong'] += (both_wrong / total_errors)
    
    # Calculate averages
    for totals in [hpc_totals, hpcdub_totals, hpce_totals]:
        for key in totals:
            totals[key] = (totals[key] / n_models) * 100  # Convert to percentage
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Colors for the pie slices (excluding the light blue for Both Correct)
    colors = ['#4477AA', '#CC6677', '#DDCC77']  # Blue, Pink/Rose, Yellow
    
    # Function to create a pie chart
    def make_pie(ax, data, title):
        labels = ['', '', '']  # Empty labels for individual pies
        sizes = [data['nc_only'], data['pc_only'], data['both_wrong']]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 18})
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(20)
        
        # Add title immediately below the pie
        ax.text(0.5, 0.05, title, fontsize=20, ha='center', transform=ax.transAxes)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
    
    # Create the three pie charts
    make_pie(ax1, hpc_totals, 'HPC')
    make_pie(ax2, hpcdub_totals, 'HPCdub')
    make_pie(ax3, hpce_totals, 'HPCE')
    
    # Create a single legend centered below the pies
    legend_labels = ['NC Only', 'PC Only', 'Both Wrong']
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i]) for i in range(3)]
    fig.legend(legend_elements, legend_labels, loc='lower center', fontsize=20, 
               bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.05, right=0.95)
    
    # Save as PDF
    output_path_pdf = os.path.join(output_dir, f'{task.lower()}_error_pie_charts.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Pie charts also saved as PDF: {output_path_pdf}")
    
    plt.close()
    
    # Print the averaged percentages
    print("\nAveraged percentages across all models (excluding Both Correct):")
    print("\nHPC:")
    print(f"  NC Only: {hpc_totals['nc_only']:.1f}%")
    print(f"  PC Only: {hpc_totals['pc_only']:.1f}%")
    print(f"  Both Wrong: {hpc_totals['both_wrong']:.1f}%")
    
    print("\nHPCdub:")
    print(f"  NC Only: {hpcdub_totals['nc_only']:.1f}%")
    print(f"  PC Only: {hpcdub_totals['pc_only']:.1f}%")
    print(f"  Both Wrong: {hpcdub_totals['both_wrong']:.1f}%")
    
    print("\nHPCE:")
    print(f"  NC Only: {hpce_totals['nc_only']:.1f}%")
    print(f"  PC Only: {hpce_totals['pc_only']:.1f}%")
    print(f"  Both Wrong: {hpce_totals['both_wrong']:.1f}%")


def create_error_strip(results: Dict[str, Dict[str, int]], output_dir: str = "results/figures", task: str = "RAG"):
    """
    Create a horizontal stacked-bar (strip) figure averaging NC Only / PC Only / Both Wrong
    percentages across models for HPC, HPCdub, and HPCE rows.

    PCK variant shows a legend on top; RAG variant adds the "Percentage (%)" x-label
    and no legend — matching doc/Paper-KnowledgeConflict/figure_files/*_error_strip.pdf.
    """
    os.makedirs(output_dir, exist_ok=True)

    hpc_totals = {'nc_only': 0.0, 'pc_only': 0.0, 'both_wrong': 0.0}
    hpcdub_totals = {'nc_only': 0.0, 'pc_only': 0.0, 'both_wrong': 0.0}
    hpce_totals = {'nc_only': 0.0, 'pc_only': 0.0, 'both_wrong': 0.0}
    n_models = len(results)

    for _, stats in results.items():
        if stats['total_hpc'] > 0:
            nc_only = stats['hpc_nc_correct_hpc_wrong']
            pc_only = stats['hpc_hpc_correct_nc_wrong']
            both_wrong = stats['hpc_both_wrong']
            tot = nc_only + pc_only + both_wrong
            if tot > 0:
                hpc_totals['nc_only'] += nc_only / tot
                hpc_totals['pc_only'] += pc_only / tot
                hpc_totals['both_wrong'] += both_wrong / tot
        if stats.get('total_hpc_double', 0) > 0:
            nc_only = stats['hpc_double_nc_correct_hpc_wrong']
            pc_only = stats['hpc_double_hpc_correct_nc_wrong']
            both_wrong = stats['hpc_double_both_wrong']
            tot = nc_only + pc_only + both_wrong
            if tot > 0:
                hpcdub_totals['nc_only'] += nc_only / tot
                hpcdub_totals['pc_only'] += pc_only / tot
                hpcdub_totals['both_wrong'] += both_wrong / tot
        if stats['total_hpce'] > 0:
            nc_only = stats['hpce_nc_correct_hpce_wrong']
            pc_only = stats['hpce_hpce_correct_nc_wrong']
            both_wrong = stats['hpce_both_wrong']
            tot = nc_only + pc_only + both_wrong
            if tot > 0:
                hpce_totals['nc_only'] += nc_only / tot
                hpce_totals['pc_only'] += pc_only / tot
                hpce_totals['both_wrong'] += both_wrong / tot

    for totals in (hpc_totals, hpcdub_totals, hpce_totals):
        for k in totals:
            totals[k] = (totals[k] / n_models) * 100

    rows = ['HPC', 'HPCdub', 'HPCE']
    row_data = [hpc_totals, hpcdub_totals, hpce_totals]
    nc = np.array([d['nc_only'] for d in row_data])
    pc = np.array([d['pc_only'] for d in row_data])
    bw = np.array([d['both_wrong'] for d in row_data])

    color_nc = '#A6C8E0'
    color_pc = '#F4A6A6'
    color_bw = '#B5D99C'

    fig, ax = plt.subplots(figsize=(12, 3.5))
    y = np.arange(len(rows))
    ax.barh(y, nc, color=color_nc, label='NC Only', edgecolor='none')
    ax.barh(y, pc, left=nc, color=color_pc, label='PC Only', edgecolor='none')
    ax.barh(y, bw, left=nc + pc, color=color_bw, label='Both Wrong', edgecolor='none')

    for i in range(len(rows)):
        if nc[i] > 3:
            ax.text(nc[i] / 2, i, f'{nc[i]:.1f}', ha='center', va='center', fontsize=22, color='black')
        if pc[i] > 3:
            ax.text(nc[i] + pc[i] / 2, i, f'{pc[i]:.1f}', ha='center', va='center', fontsize=22, color='black')
        if bw[i] > 3:
            ax.text(nc[i] + pc[i] + bw[i] / 2, i, f'{bw[i]:.1f}', ha='center', va='center', fontsize=22, color='black')

    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=22)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='x', labelsize=18)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    if task.upper() == 'PCK':
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=3, frameon=False, fontsize=20, handlelength=1.5, handleheight=1.2)
    else:
        ax.set_xlabel('Percentage (%)', fontsize=22)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{task.lower()}_error_strip.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Strip plot saved as PDF: {out_path}")
    plt.close()


if __name__ == "__main__":
    import sys

    # Get task type from command line argument, default to RAG
    task = sys.argv[1] if len(sys.argv) > 1 else "RAG"

    # Analyze all models
    results = analyze_all_models(task=task)

    # Print results
    print_error_analysis(results, task=task)

    # Print table
    print_error_table(results)

    # Create visualization
    visualize_error_analysis(results, task=task)

    # Create pie charts
    create_pie_charts(results, task=task)

    # Create strip plot
    create_error_strip(results, task=task)