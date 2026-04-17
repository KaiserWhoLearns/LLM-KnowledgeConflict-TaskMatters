#!/bin/bash
# Extract a model's parametric knowledge on the evaluation questions.
#
# Configure via env vars or edit the defaults below:
#   MODEL_NAME  HuggingFace repo id or API model id
set -e

export base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export data_dir="${base_dir}/data"

export model_name="${MODEL_NAME:-gpt-5.2}"

reasoning_flag=""
if [[ "$model_name" == "Qwen/Qwen3-8B" || "$model_name" == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ]]; then
    reasoning_flag="--reasoning_model"
fi

declare -A MODEL_NAME_TO_PRETTY
MODEL_NAME_TO_PRETTY["Qwen/Qwen3-8B"]="qwen3-8B"
MODEL_NAME_TO_PRETTY["google/gemma-3-4b-it"]="gemma3-4b"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-7B-Instruct"]="olmo2-7B"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-13B-Instruct"]="olmo2-13B"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-1B-Instruct"]="llama3.2-1B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-3B-Instruct"]="llama3.2-3B-Instruct"
MODEL_NAME_TO_PRETTY["mistralai/Mistral-7B-Instruct-v0.3"]="mistral7B"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-7B-Instruct-1M"]="qwen7B-instruct"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5-14B-instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.3-70B-Instruct"]="llama3.3-70B-Instruct"
MODEL_NAME_TO_PRETTY["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-llama8b"
MODEL_NAME_TO_PRETTY["gpt-5.2"]="gpt5.2"

pretty_name="${MODEL_NAME_TO_PRETTY[$model_name]}"

echo "Running ${pretty_name}-get_knowledge"
mkdir -p "${base_dir}/logs"
cd "$base_dir"

python data_creation/get_parametric_knowledge.py \
    $reasoning_flag \
    --model_name "$model_name"
