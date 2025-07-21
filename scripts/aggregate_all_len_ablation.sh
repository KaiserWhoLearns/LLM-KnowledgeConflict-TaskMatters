#!/bin/bash

# Script to run aggregate_eval_results.py for all available models in length ablation format
# This script identifies all models that have mult_perf.csv files and runs length ablation aggregation

cd "$(dirname "$0")/.."

echo "Running length ablation aggregation for all available models..."

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