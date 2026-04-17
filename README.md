# Knowledge Conflict: Task Matters

Code and data pipeline for evaluating how large language models handle conflicts
between parametric and contextual knowledge across different task formulations
(knowledge-free, parametric-only, contextual-only, parametric+contextual, and RAG).

## Setup

```bash
# Create the conda environment
conda env create -f environment.yml
conda activate kc

# Provide API credentials (required for GPT-5.2 / Together models)
cp .env.example .env   # or create .env manually
# then edit .env and set:
#   OPENAI_API_KEY=...
#   TOGETHER_API_KEY=...     # optional, only if using Together models
```

For gated HuggingFace checkpoints (e.g. Llama), log in once with
`huggingface-cli login`.

All paths default to the repository root. Override them via environment
variables if you need a different data or cache location:

```bash
export base_dir=$(pwd)
export data_dir=$base_dir/data
export HF_HOME=$base_dir/hf_cache
```

## Data

The raw source datasets are loaded from HuggingFace:

```python
from datasets import load_dataset
load_dataset("ibm-research/Wikipedia_contradict_benchmark", split="train")
load_dataset("copenlu/druid", split="train")
load_dataset("KaiserWhoLearns/conflictqa-u", split="train")
```

`data/conflictqa.pkl` is included for convenience; `utils/data_utils.py`
converts it to a HuggingFace dataset and merges it with WikiContradict.

## Pipeline

All scripts are plain bash. Each script reads `MODEL_NAME`, `TASK_TYPE`, and
`DATA_VERSION` from environment variables (defaults shown in the script
header). Run them from any directory — they resolve paths relative to the
repo root.

### 1. Get parametric knowledge of a model

```bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" bash scripts/get_parametric_knowledge.sh
```

Produces `data/model_knowledge/<pretty_name>/` used downstream to select
questions on which the model has correct parametric knowledge.

### 2. Build task datasets (context editing)

```bash
MODEL_NAME="gpt-5.2" bash scripts/context_edit.sh
```

Runs, in order:
`data_creation/clas_edit_context.py` → `remove_invalid_instances.py` →
`add_instruction.py` → `add_instruction_choice.py`.
Final outputs land in `data/task_data/` (free-form) and `data/choice_task/`
(multiple choice).

### 3. Run predictions + evaluation

Multiple choice:
```bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" TASK_TYPE=PK bash scripts/pred_choice.sh
```

Free-form generation:
```bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" TASK_TYPE=KFextract bash scripts/pred_free.sh
```

Supported `TASK_TYPE`: `KFsummary`, `KFextract`, `PCK`, `CK`, `PK`, `RAG`.

Ablations (length / prompt strength) are toggled via env vars:
```bash
LENGTH_ABLATION=true bash scripts/pred_choice.sh
PROMPT_ABLATION=true bash scripts/pred_choice.sh
```

### 4. Aggregate results

```bash
MODEL_NAME=mistral7B bash scripts/get_results.sh
bash scripts/aggregate_ablation_res.sh --len      # or --prompt
```

CSVs are written to `results/`.

## Repository layout

```
data_creation/   Scripts that build task datasets from raw sources
model_runs/      predict.py, evaluate.py, evaluate_choice.py,
                 aggregate_eval_results.py
scripts/         Bash entry points for the pipeline above
analysis/        Plotting and error-analysis scripts (figures in results/)
utils/           Path constants and dataset-merge helpers
prompts/         Task instruction templates
results/         Aggregated CSVs and paper figures
```

## Supported models

Pretty names ↔ HuggingFace IDs are defined in `model_runs/predict.py` and the
shell scripts. Out of the box: OLMo-2 (7B/13B), Llama-3.1/3.2, Mistral-7B,
Qwen2.5 (7B/14B), DeepSeek-R1-Distill-Llama-8B, Gemma-3-4B, and OpenAI
`gpt-5.2`.
