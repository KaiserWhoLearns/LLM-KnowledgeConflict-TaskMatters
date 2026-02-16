#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct
export data_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct/data

export model_name="gpt-5.2"
# export model_name="Qwen/Qwen2.5-14B-Instruct"
# export task_type="RAG"
export data_version="full_v2"
export length_ablation=false # Set to true for length ablation experiments
export prompt_ablation=true # Set to true for prompt ablation experiments (creates all: weak, neutral, strong)

# declare -A TASK_TYPE_PRETTY
# TASK_TYPE_PRETTY["KFsummary"]="knowledge_free_summary"
# TASK_TYPE_PRETTY["KFextract"]="knowledge_free_extract"
# TASK_TYPE_PRETTY["PCK"]="parametriccontextual_knowledge"
# TASK_TYPE_PRETTY["CK"]="contextual_knowledge"
# TASK_TYPE_PRETTY["PK"]="parametric_knowledge"
# TASK_TYPE_PRETTY["RAG"]="rag"
declare -A MODEL_NAME_TO_PRETTY
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-7B-Instruct"]="olmo2-7B"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-13B-Instruct"]="olmo2-13B"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.1-8B-Instruct"]="llama-3.1-8B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-1B-Instruct"]="llama3.2-1B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-3B-Instruct"]="llama3.2-3B-Instruct"
MODEL_NAME_TO_PRETTY["mistralai/Mistral-7B-Instruct-v0.3"]="mistral7B"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-7B-Instruct-1M"]="qwen7B-instruct"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5-14B-instruct"
MODEL_NAME_TO_PRETTY["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-llama8b"
MODEL_NAME_TO_PRETTY["gpt-5.2"]="gpt5.2"

# Determine if this is an API model (no GPU needed)
is_api_model=false
if [[ "$model_name" == gpt-* ]]; then
    is_api_model=true
fi

export exp_name="${MODEL_NAME_TO_PRETTY[$model_name]}-add-instruct"
echo "Running $exp_name"

# Check mutual exclusion
if [ "$length_ablation" = true ] && [ "$prompt_ablation" = true ]; then
    echo "Error: length_ablation and prompt_ablation cannot both be true"
    exit 1
fi

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
#SBATCH --time=1-15:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
cd $base_dir

# Determine which script to run based on ablation settings
if [ "$length_ablation" = true ]; then
    echo "Running length ablation experiments"
    python data_creation/length_ablation.py \
        --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
        --data_version $data_version
elif [ "$prompt_ablation" = true ]; then
    echo "Running prompt ablation experiments for all strength levels"
    # Run for all prompt types: weak, neutral, strong
    for prompt_strength in weak neutral strong; do
        echo "Creating data with \$prompt_strength prompts"
        python data_creation/prompt_ablation.py \
            --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
            --data_version $data_version \
            --prompt_type \$prompt_strength \
            --task_type all
    done
else
    echo "Running standard instruction addition"
    python data_creation/add_instruction_choice.py \
        --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
        --data_version $data_version
fi

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
#SBATCH --gpus=1
#SBATCH --time=1-15:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
module load cuda/11.8.0
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
# source "/home/hsun74/.bashrc"
cd $base_dir

# Determine which script to run based on ablation settings
if [ "$length_ablation" = true ]; then
    echo "Running length ablation experiments"
    python data_creation/length_ablation.py \
        --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
        --data_version $data_version
elif [ "$prompt_ablation" = true ]; then
    echo "Running prompt ablation experiments for all strength levels"
    # Run for all prompt types: weak, neutral, strong
    for prompt_strength in weak neutral strong; do
        echo "Creating data with \$prompt_strength prompts"
        python data_creation/prompt_ablation.py \
            --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
            --data_version $data_version \
            --prompt_type \$prompt_strength \
            --task_type all
    done
else
    echo "Running standard instruction addition"
    python data_creation/add_instruction_choice.py \
        --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
        --data_version $data_version
fi

EOT
fi