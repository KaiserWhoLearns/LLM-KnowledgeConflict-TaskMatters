# Exp Notes

### Load dataset shorthand

Everything is in Huggingface, can load them directly.
```
dataset = load_dataset("ibm/Wikipedia_contradict_benchmark", split="train")
dataset = load_dataset("copenlu/druid", split="train")
dataset = load_dataset("KaiserWhoLearns/conflictqa-u", split="train")
```



### Preliminary Experiments

1. Test with synthetic data edit

2. Test parametric knowledge


module load anaconda
conda activate /scratch4/mdredze1/hsun74/conda_env/kc