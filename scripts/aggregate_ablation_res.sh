#!/bin/bash

# Script to run aggregate_eval_results.py for all available models in ablation formats
# This script can aggregate either length ablation or prompt ablation results

cd "$(dirname "$0")/.."

# Parse command line arguments
ABLATION_TYPE="len"

usage() {
    echo "Usage: $0 [--len|--prompt]"
    echo "  --len     Aggregate length ablation results"
    echo "  --prompt  Aggregate prompt ablation results"
    exit 1
}

# Check for arguments
if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    --len)
        ABLATION_TYPE="len"
        ;;
    --prompt)
        ABLATION_TYPE="prompt"
        ;;
    *)
        echo "Error: Invalid option '$1'"
        usage
        ;;
esac

# Array of models found in results directory
models=(
    "mistral7B"
    "olmo2-7B"
    "olmo2-13B"
    "qwen7B-instruct"
    "qwen2.5-14B-instruct"
)

# Data version and format
data_version="full_v2"
format="mult"

if [ "$ABLATION_TYPE" = "len" ]; then
    echo "Running length ablation aggregation for all available models..."
    
    for model in "${models[@]}"; do
        echo "Processing model: $model"
        python model_runs/aggregate_eval_results.py \
            --test_model_name "$model" \
            --data_version "$data_version" \
            --format "$format" \
            --len_ablation
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed $model"
        else
            echo "✗ Failed to process $model"
        fi
        echo "---"
    done
    
    echo "Length ablation aggregation complete!"
    echo "Results saved to results/ directory with *_len_ablation.csv suffix"

elif [ "$ABLATION_TYPE" = "prompt" ]; then
    echo "Running prompt ablation aggregation for all available models..."
    
    for model in "${models[@]}"; do
        echo "Processing model: $model"
        python model_runs/aggregate_eval_results.py \
            --test_model_name "$model" \
            --data_version "$data_version" \
            --format "$format" \
            --prompt_ablation
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed $model"
        else
            echo "✗ Failed to process $model"
        fi
        echo "---"
    done
    
    echo "Prompt ablation aggregation complete!"
    echo "Results saved to results/ directory with *_prompt_ablation.csv suffix"
fi