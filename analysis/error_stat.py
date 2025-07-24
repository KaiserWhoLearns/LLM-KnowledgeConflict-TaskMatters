import json
import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def count_error_types(model_name: str, base_dir: str = "/scratch4/mdredze1/hsun74/KnowledgeInstruct/output", include_len_ablation: bool = True) -> Dict[str, int]:
    """
    Count error types for RAG tasks by comparing predictions with correct answers.
    
    For each context type (NC, HPC, HPCE, HPC double), we track:
    1. Get NC part correct but HPC/HPCE part wrong
    2. Get HPC/HPCE part correct but NC part wrong
    3. Get both parts wrong
    4. Get both parts correct
    
    Args:
        model_name: Name of the model (e.g., "mistral7B", "olmo2-7B")
        base_dir: Base directory containing output files
        include_len_ablation: Whether to include HPC double from len_ablation files
        
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
    predictions_file = os.path.join(base_dir, "metrics_mult", f"{model_name}_RAG_full_v2_choice.jsonl")
    
    # Read ground truth from model RAG file
    ground_truth_file = os.path.join(base_dir, f"{model_name}_RAG_full_v2_choice.jsonl")
    
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
        len_ablation_predictions_file = os.path.join(base_dir, "metrics_mult", f"{model_name}_RAG_full_v2_choice_len_ablation.jsonl")
        len_ablation_ground_truth_file = os.path.join(base_dir, f"{model_name}_RAG_full_v2_choice_len_ablation.jsonl")
        
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


def analyze_all_models(base_dir: str = "/scratch4/mdredze1/hsun74/KnowledgeInstruct/output") -> Dict[str, Dict[str, int]]:
    """
    Analyze error types for all models with RAG results.
    
    Args:
        base_dir: Base directory containing output files
        
    Returns:
        Dictionary mapping model names to their error statistics
    """
    # Find all models with RAG results
    metrics_dir = os.path.join(base_dir, "metrics_mult")
    models = set()
    
    for filename in os.listdir(metrics_dir):
        if filename.endswith("_RAG_full_v2_choice.jsonl") and not filename.endswith("_len_ablation.jsonl"):
            model_name = filename.replace("_RAG_full_v2_choice.jsonl", "")
            models.add(model_name)
    
    # Analyze each model
    results = {}
    for model in sorted(models):
        try:
            results[model] = count_error_types(model, base_dir)
        except Exception as e:
            print(f"Error processing {model}: {e}")
    
    return results


def print_error_analysis(results: Dict[str, Dict[str, int]]):
    """
    Print formatted error analysis results.
    
    Args:
        results: Dictionary mapping model names to error statistics
    """
    print("\nRAG Task Error Analysis")
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


def visualize_error_analysis(results: Dict[str, Dict[str, int]], output_dir: str = "results/figures"):
    """
    Create a stacked percentage bar plot for error analysis showing only NC Only and HPC/HPCE Only.
    Now includes HPC, HPCE, and HPC double bars.
    
    Args:
        results: Dictionary mapping model names to error statistics
        output_dir: Directory to save the figure
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
        'qwen2.5-14B-instruct': 'Qwen-14B'
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
                    label='HPC Only', color=hpc_color)
    
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
    
    # Save figure
    output_path = os.path.join(output_dir, 'rag_error_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = os.path.join(output_dir, 'rag_error_analysis.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Figure also saved as PDF: {output_path_pdf}")
    
    plt.close()


if __name__ == "__main__":
    # Analyze all models
    results = analyze_all_models()
    
    # Print results
    print_error_analysis(results)
    
    # Print table
    print_error_table(results)
    
    # Create visualization
    visualize_error_analysis(results)