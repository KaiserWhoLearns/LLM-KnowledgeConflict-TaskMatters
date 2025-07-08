#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct
export data_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct/data

export model_name="Qwen/Qwen2.5-14B-Instruct"
data_version="full_v2"

declare -A MODEL_NAME_TO_PRETTY 
# No examples captured for Gemma3-4b
MODEL_NAME_TO_PRETTY["google/gemma-3-4b-it"]="gemma3-4b"

MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-7B-Instruct"]="olmo2-7B"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-13B-Instruct"]="olmo2-13B"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.1-8B-Instruct"]="llama-3.1-8B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-1B-Instruct"]="llama3.2-1B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-3B-Instruct"]="llama3.2-3B-Instruct"
MODEL_NAME_TO_PRETTY["mistralai/Mistral-7B-Instruct-v0.3"]="mistral7B"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5-14B-instruct"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-7B-Instruct-1M"]="qwen7B-instruct"
MODEL_NAME_TO_PRETTY["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-llama8b" 

export exp_name="${MODEL_NAME_TO_PRETTY[$model_name]}-context_edit"
echo "Running $exp_name"

export BNB_CUDA_VERSION=118

mkdir -p ${base_dir}/logs

cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -A mdredze80_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
module load cuda/11.8.0
module load git-lfs
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
# source "/home/hsun74/.bashrc"
cd $base_dir

python data_creation/clas_edit_context.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --data_version $data_version \
    --use_batch

python data_creation/remove_invalid_instances.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --data_version $data_version


python data_creation/add_instruction.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --data_version $data_version

EOT