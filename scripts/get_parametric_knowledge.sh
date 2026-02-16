#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct
export data_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct/data

export model_name="gpt-5.2"
# export model_name="Qwen/Qwen2.5-14B-Instruct"
# export model_name="allenai/OLMo-2-1124-13B-Instruct"

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
MODEL_NAME_TO_PRETTY["Qwen/Qwen3-8B"]="qwen3-8B"
MODEL_NAME_TO_PRETTY["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-llama8b"
MODEL_NAME_TO_PRETTY["gpt-5.2"]="gpt5.2"

# Determine if this is an API model (no GPU needed)
is_api_model=false
if [[ "$model_name" == gpt-* ]]; then
    is_api_model=true
fi

export exp_name="${MODEL_NAME_TO_PRETTY[$model_name]}-get_knoweldge-2gpu"
echo "Running $exp_name"

export BNB_CUDA_VERSION=118

mkdir -p ${base_dir}/logs

cd $base_dir

if [ "$is_api_model" = true ]; then
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=parallel
#SBATCH -A mdredze1
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3/2022.05
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
cd $base_dir

python data_creation/get_parametric_knowledge.py \
    --model_name $model_name

EOT
else
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=a100
#SBATCH -A mdredze1_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gpus=3
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3/2022.05
module load cuda/11.8.0
module load git-lfs
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
# source "/home/hsun74/.bashrc"
cd $base_dir

python data_creation/get_parametric_knowledge.py \
    $reasoning_flag \
    --model_name $model_name

EOT
fi