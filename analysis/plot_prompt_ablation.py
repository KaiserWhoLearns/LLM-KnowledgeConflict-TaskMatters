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
    parser.add_argument("--all", action="store_true", help="Plot all models")
    parser.add_argument("--compare", action="store_true", help="Create comparison plot")
    parser.add_argument("--data_version", type=str, default="full_v2", help="Data version")
    parser.add_argument("--output_dir", type=str, default="figures", help="Output directory for plots")
    
    args = parser.parse_args()
    
    if args.compare:
        create_comparison_plot(data_version=args.data_version, output_dir=args.output_dir)
    elif args.all:
        plot_all_models_prompt_ablation(data_version=args.data_version, output_dir=args.output_dir)
    elif args.model:
        plot_prompt_ablation_bars(args.model, data_version=args.data_version, output_dir=args.output_dir)
    else:
        # Default: plot all models and comparison
        plot_all_models_prompt_ablation(data_version=args.data_version, output_dir=args.output_dir)
        create_comparison_plot(data_version=args.data_version, output_dir=args.output_dir)