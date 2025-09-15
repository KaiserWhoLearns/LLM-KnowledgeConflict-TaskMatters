import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_prompt_ablation_data(model_name, data_version="full_v2"):
    """
    Load prompt ablation results from CSV files
    """
    base_dir = os.environ.get("base_dir", ".")
    csv_path = os.path.join(base_dir, "results", f"{model_name}_{data_version}_mult_perf_prompt_ablation.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def get_metric_for_task(task_type):
    """
    Return the appropriate metric for each task type
    """
    if task_type in ["PCK", "RAG"]:
        return "f1"
    elif task_type in ["PK", "CK"]:
        return "exact_match"
    else:
        return "f1"  # default

def plot_prompt_ablation_bars(model_name, data_version="full_v2", output_dir="figures"):
    """
    Create bar plots for prompt ablation results
    Each plot shows performance across weak, neutral, strong prompts for each evidence type
    """
    # Load data
    df = load_prompt_ablation_data(model_name, data_version)
    if df is None:
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define task types and evidence types
    task_types = ["CK", "PK", "PCK", "RAG"]
    evidence_types = ["NC", "HPC", "HPCE", "LPC"]
    prompt_strengths = ["weak", "neutral", "strong"]
    
    # Create a figure with subplots for each task
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Prompt Ablation Results - {model_name}', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Colors for each prompt strength
    colors = {'weak': '#FF6B6B', 'neutral': '#4ECDC4', 'strong': '#45B7D1'}
    
    for idx, task_type in enumerate(task_types):
        ax = axes[idx]
        
        # Get the appropriate metric for this task
        metric_name = get_metric_for_task(task_type)
        
        # Filter data for this task and metric
        task_data = df[(df['task'] == task_type) & (df['metric'] == metric_name)]
        
        if task_data.empty:
            ax.text(0.5, 0.5, f'No data for {task_type}', ha='center', va='center')
            ax.set_title(f'{task_type} - {metric_name}')
            continue
        
        # Prepare data for plotting
        bar_data = []
        bar_labels = []
        bar_colors = []
        
        x_positions = []
        x_labels = []
        
        for i, evidence_type in enumerate(evidence_types):
            for j, strength in enumerate(prompt_strengths):
                col_name = f"{evidence_type}-{strength}"
                if col_name in task_data.columns:
                    value = task_data[col_name].values[0]
                    if not pd.isna(value):
                        position = i * (len(prompt_strengths) + 0.5) + j
                        x_positions.append(position)
                        bar_data.append(value)
                        bar_colors.append(colors[strength])
                        
                        # Only add evidence type label for the middle bar
                        if j == 1:
                            x_labels.append(evidence_type)
        
        # Create bar plot
        bars = ax.bar(x_positions, bar_data, color=bar_colors, width=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Set x-axis labels at group centers
        group_centers = [i * (len(prompt_strengths) + 0.5) + 1 for i in range(len(evidence_types))]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(evidence_types)
        
        # Set labels and title
        ax.set_xlabel('Evidence Type', fontweight='bold')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()} (%)', fontweight='bold')
        ax.set_title(f'{task_type} - {metric_name.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylim(0, 105)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[s], edgecolor='black', linewidth=0.5, label=s.capitalize()) 
                      for s in prompt_strengths]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), title='Prompt Strength')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    output_file = output_path / f"{model_name}_prompt_ablation_bars.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    # Also save as PDF for publication
    output_file_pdf = output_path / f"{model_name}_prompt_ablation_bars.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved plot to {output_file_pdf}")
    
    plt.show()
    plt.close()

def plot_all_models_prompt_ablation(models=None, data_version="full_v2", output_dir="figures"):
    """
    Generate prompt ablation plots for all specified models
    """
    if models is None:
        # Default model list
        models = [
            "mistral7B",
            "olmo2-7B", 
            "olmo2-13B",
            "qwen7B-instruct",
            "qwen2.5-14B-instruct"
        ]
    
    for model in models:
        print(f"\nProcessing {model}...")
        try:
            plot_prompt_ablation_bars(model, data_version, output_dir)
        except Exception as e:
            print(f"Error processing {model}: {e}")

def plot_prompt_ablation_average(model_name, data_version="full_v2", output_dir="figures"):
    """
    Create a single bar plot for a model showing average performance across all tasks and evidence types
    with standard error bars for each prompt strength
    """
    # Load data
    df = load_prompt_ablation_data(model_name, data_version)
    if df is None:
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    task_types = ["CK", "PK", "PCK", "RAG"]
    evidence_types = ["NC", "HPC", "HPCE", "LPC"]
    prompt_strengths = ["weak", "neutral", "strong"]
    
    # Collect all values for each prompt strength
    strength_values = {strength: [] for strength in prompt_strengths}
    
    for task_type in task_types:
        # Get the appropriate metric for this task
        metric_name = get_metric_for_task(task_type)
        
        # Filter data for this task and metric
        task_data = df[(df['task'] == task_type) & (df['metric'] == metric_name)]
        
        if task_data.empty:
            continue
        
        for evidence_type in evidence_types:
            for strength in prompt_strengths:
                col_name = f"{evidence_type}-{strength}"
                if col_name in task_data.columns:
                    value = task_data[col_name].values[0]
                    if not pd.isna(value):
                        strength_values[strength].append(value)
    
    # Calculate means and standard errors
    means = []
    stderrs = []
    labels = []
    
    for strength in prompt_strengths:
        if strength_values[strength]:
            values = np.array(strength_values[strength])
            means.append(np.mean(values))
            stderrs.append(np.std(values) / np.sqrt(len(values)))  # Standard error
            labels.append(strength.capitalize())
        else:
            means.append(0)
            stderrs.append(0)
            labels.append(strength.capitalize())
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors for each prompt strength
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # weak, neutral, strong
    
    # Create bar plot with error bars
    x_pos = np.arange(len(prompt_strengths))
    bars = ax.bar(x_pos, means, yerr=stderrs, capsize=5, 
                  color=colors, edgecolor='black', linewidth=1.5,
                  error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    # Add value labels on bars
    for i, (bar, mean, stderr) in enumerate(zip(bars, means, stderrs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stderr,
               f'{mean:.1f}±{stderr:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Prompt Strength', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Prompt Ablation Results - {model_name}\n(Averaged across all tasks and evidence types)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.2 if means else 100)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add sample size annotation
    total_samples = sum(len(strength_values[s]) for s in prompt_strengths)
    ax.text(0.02, 0.98, f'n = {total_samples} total measurements', 
           transform=ax.transAxes, fontsize=9, va='top')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f"{model_name}_prompt_ablation_average.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved average plot to {output_file}")
    
    # Also save as PDF
    output_file_pdf = output_path / f"{model_name}_prompt_ablation_average.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved average plot to {output_file_pdf}")
    
    plt.show()
    plt.close()

def plot_all_models_average(models=None, data_version="full_v2", output_dir="figures"):
    """
    Create average performance plots for all models in a grid layout
    """
    if models is None:
        # models = ["mistral7B", "olmo2-7B", "olmo2-13B", "qwen7B-instruct", "qwen2.5-14B-instruct"]
        models = ["mistral7B", "olmo2-7B", "olmo2-13B", "qwen7B-instruct", "qwen2.5-14B-instruct"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter to only models with data
    valid_models = []
    for model in models:
        df = load_prompt_ablation_data(model, data_version)
        if df is not None:
            valid_models.append(model)
    
    if not valid_models:
        print("No valid models found with data")
        return
    
    # Create subplot grid
    n_models = len(valid_models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Prompt Ablation Average Performance Across Models', fontsize=16, fontweight='bold')
    
    task_types = ["CK", "PK", "PCK", "RAG"]
    evidence_types = ["NC", "HPC", "HPCE", "LPC"]
    prompt_strengths = ["weak", "neutral", "strong"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, model in enumerate(valid_models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        df = load_prompt_ablation_data(model, data_version)
        
        # Collect all values for each prompt strength
        strength_values = {strength: [] for strength in prompt_strengths}
        
        for task_type in task_types:
            metric_name = get_metric_for_task(task_type)
            task_data = df[(df['task'] == task_type) & (df['metric'] == metric_name)]
            
            if task_data.empty:
                continue
            
            for evidence_type in evidence_types:
                for strength in prompt_strengths:
                    col_name = f"{evidence_type}-{strength}"
                    if col_name in task_data.columns:
                        value = task_data[col_name].values[0]
                        if not pd.isna(value):
                            strength_values[strength].append(value)
        
        # Calculate means and standard errors
        means = []
        stderrs = []
        
        for strength in prompt_strengths:
            if strength_values[strength]:
                values = np.array(strength_values[strength])
                means.append(np.mean(values))
                stderrs.append(np.std(values) / np.sqrt(len(values)))
            else:
                means.append(0)
                stderrs.append(0)
        
        # Create bar plot
        x_pos = np.arange(len(prompt_strengths))
        bars = ax.bar(x_pos, means, yerr=stderrs, capsize=5,
                      color=colors, edgecolor='black', linewidth=1,
                      error_kw={'linewidth': 1, 'ecolor': 'black'})
        
        # Add value labels
        for bar, mean, stderr in zip(bars, means, stderrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stderr,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Prompt Strength', fontsize=10)
        ax.set_ylabel('Avg Performance (%)', fontsize=10)
        ax.set_title(model.replace("-instruct", ""), fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.capitalize() for s in prompt_strengths])
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save figure
    output_file = output_path / "all_models_prompt_ablation_average_grid.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved grid plot to {output_file}")
    
    output_file_pdf = output_path / "all_models_prompt_ablation_average_grid.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved grid plot to {output_file_pdf}")
    
    plt.show()
    plt.close()

def compute_aggregate_prompt_statistics(models=None, data_version="full_v2"):
    """
    Compute aggregate average and standard deviation for each prompt strength (weak, neutral, strong):
    1. For each individual model (averaged across tasks and evidence types)
    2. Across all models, tasks, and evidence types
    
    Returns:
        tuple: (per_model_stats, aggregate_stats)
            - per_model_stats: dict with model names as keys and dict of prompt strength stats as values
            - aggregate_stats: dict with prompt strengths as keys and tuples of (mean, std) as values
    """
    if models is None:
        models = ["mistral7B", "olmo2-7B", "olmo2-13B", "qwen7B-instruct", "qwen2.5-14B-instruct"]
    
    task_types = ["CK", "PK", "PCK", "RAG"]
    evidence_types = ["NC", "HPC", "HPCE", "LPC"]
    prompt_strengths = ["weak", "neutral", "strong"]
    
    # Store per-model statistics
    per_model_stats = {}
    
    # Collect all values for each prompt strength across all models
    all_values = {strength: [] for strength in prompt_strengths}
    
    for model in models:
        # Load data for this model
        df = load_prompt_ablation_data(model, data_version)
        if df is None:
            continue
        
        # Collect values for this specific model
        model_values = {strength: [] for strength in prompt_strengths}
        
        # Iterate through all tasks
        for task_type in task_types:
            # Get the appropriate metric for this task
            metric_name = get_metric_for_task(task_type)
            
            # Filter data for this task and metric
            task_data = df[(df['task'] == task_type) & (df['metric'] == metric_name)]
            
            if task_data.empty:
                continue
            
            # Iterate through all evidence types
            for evidence_type in evidence_types:
                # Collect values for each prompt strength
                for strength in prompt_strengths:
                    col_name = f"{evidence_type}-{strength}"
                    if col_name in task_data.columns:
                        value = task_data[col_name].values[0]
                        if not pd.isna(value):
                            model_values[strength].append(value)
                            all_values[strength].append(value)
        
        # Compute statistics for this model
        model_stats = {}
        for strength in prompt_strengths:
            if model_values[strength]:
                values = np.array(model_values[strength])
                mean_val = np.mean(values)
                std_val = np.std(values)
                model_stats[strength] = (mean_val, std_val)
            else:
                model_stats[strength] = (0.0, 0.0)
        
        per_model_stats[model] = model_stats
    
    # Print per-model statistics
    print("\n" + "="*70)
    print("PER-MODEL STATISTICS (Averaged across tasks and evidence types):")
    print("="*70)
    
    for model in per_model_stats:
        print(f"\n{model}:")
        print("-" * 40)
        for strength in prompt_strengths:
            if strength in per_model_stats[model]:
                mean_val, std_val = per_model_stats[model][strength]
                print(f"  {strength.capitalize():8} - Mean: {mean_val:6.2f}% ± {std_val:5.2f}%")
    
    # Compute aggregate statistics across all models
    aggregate_stats = {}
    
    print("\n" + "="*70)
    print("DETAILED AGGREGATE STATISTICS:")
    print("="*70)
    
    for strength in prompt_strengths:
        if all_values[strength]:
            values = np.array(all_values[strength])
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregate_stats[strength] = (mean_val, std_val)
            
            # Print detailed statistics
            print(f"\n{strength.upper()} Prompt Statistics:")
            print(f"  Mean: {mean_val:.2f}%")
            print(f"  Std Dev: {std_val:.2f}%")
            print(f"  Min: {np.min(values):.2f}%")
            print(f"  Max: {np.max(values):.2f}%")
            print(f"  Median: {np.median(values):.2f}%")
            print(f"  Sample Size: {len(values)}")
        else:
            aggregate_stats[strength] = (0.0, 0.0)
            print(f"\n{strength.upper()} Prompt Statistics: No data available")
    
    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY - AGGREGATE ACROSS ALL MODELS, TASKS, AND EVIDENCE TYPES:")
    print("="*70)
    for strength in prompt_strengths:
        if strength in aggregate_stats:
            mean_val, std_val = aggregate_stats[strength]
            print(f"{strength.capitalize():8} - Mean: {mean_val:6.2f}% ± {std_val:5.2f}%")
    
    return per_model_stats, aggregate_stats

def create_comparison_plot(models=None, data_version="full_v2", output_dir="figures"):
    """
    Create a comparison plot showing all models' performance across prompt strengths
    """
    if models is None:
        models = ["mistral7B", "olmo2-7B", "olmo2-13B", "qwen7B-instruct", "qwen2.5-14B-instruct"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    task_types = ["CK", "PK", "PCK", "RAG"]
    prompt_strengths = ["weak", "neutral", "strong"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prompt Ablation Comparison Across Models', fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for idx, task_type in enumerate(task_types):
        ax = axes[idx]
        metric_name = get_metric_for_task(task_type)
        
        # Prepare data for grouped bar plot
        data_dict = {strength: [] for strength in prompt_strengths}
        model_labels = []
        
        for model in models:
            df = load_prompt_ablation_data(model, data_version)
            if df is None:
                continue
                
            task_data = df[(df['task'] == task_type) & (df['metric'] == metric_name)]
            if task_data.empty:
                continue
            
            model_labels.append(model.replace("-instruct", ""))
            
            # Calculate average across all evidence types for each strength
            for strength in prompt_strengths:
                values = []
                for evidence_type in ["NC", "HPC", "HPCE", "LPC"]:
                    col_name = f"{evidence_type}-{strength}"
                    if col_name in task_data.columns:
                        value = task_data[col_name].values[0]
                        if not pd.isna(value):
                            values.append(value)
                
                avg_value = np.mean(values) if values else 0
                data_dict[strength].append(avg_value)
        
        if not model_labels:
            ax.text(0.5, 0.5, f'No data for {task_type}', ha='center', va='center')
            ax.set_title(f'{task_type}')
            continue
        
        # Create grouped bar plot
        x = np.arange(len(model_labels))
        width = 0.25
        
        colors = {'weak': '#FF6B6B', 'neutral': '#4ECDC4', 'strong': '#45B7D1'}
        
        for i, strength in enumerate(prompt_strengths):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, data_dict[strength], width, 
                         label=strength.capitalize(), color=colors[strength],
                         edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()} (%)', fontweight='bold')
        ax.set_title(f'{task_type} - {metric_name.replace("_", " ").title()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=45, ha='right')
        ax.legend(title='Prompt Strength', loc='upper left')
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save figure
    output_file = output_path / "all_models_prompt_ablation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_file}")
    
    output_file_pdf = output_path / "all_models_prompt_ablation_comparison.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Saved comparison plot to {output_file_pdf}")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot prompt ablation results")
    parser.add_argument("--model", type=str, help="Model name to plot (e.g., mistral7B)")
    parser.add_argument("--all", action="store_true", help="Plot all models (detailed)")
    parser.add_argument("--average", action="store_true", help="Plot single model with average across tasks/evidence")
    parser.add_argument("--all_average", action="store_true", help="Plot all models with averages in grid")
    parser.add_argument("--compare", action="store_true", help="Create comparison plot")
    parser.add_argument("--aggregate_stats", action="store_true", help="Compute aggregate statistics across all models")
    parser.add_argument("--data_version", type=str, default="full_v2", help="Data version")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if args.aggregate_stats:
        compute_aggregate_prompt_statistics(data_version=args.data_version)
    elif args.compare:
        create_comparison_plot(data_version=args.data_version, output_dir=args.output_dir)
    elif args.all:
        plot_all_models_prompt_ablation(data_version=args.data_version, output_dir=args.output_dir)
    elif args.all_average:
        plot_all_models_average(data_version=args.data_version, output_dir=args.output_dir)
    elif args.average and args.model:
        plot_prompt_ablation_average(args.model, data_version=args.data_version, output_dir=args.output_dir)
    elif args.model:
        plot_prompt_ablation_bars(args.model, data_version=args.data_version, output_dir=args.output_dir)
    else:
        # Default: plot all models detailed and comparison
        plot_all_models_prompt_ablation(data_version=args.data_version, output_dir=args.output_dir)
        create_comparison_plot(data_version=args.data_version, output_dir=args.output_dir)