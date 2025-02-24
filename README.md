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


{'question': 'Which of the following are present in Nymphaea nouchali var. caerulea: apomorphine, aporphine, or neither?', 'NC_context': 'Apomorphine is said to be main psychoactive compound present.', 'NC_answer': 'Apomorphine', 'alt_answer': 'Aporphine', 'alt_context': 'Like other species in the genus, Nymphaea nouchali var. caerulea contains the psychoactive alkaloid aporphine (not to be confused with apomorphine, a metabolic product of aporphine).', 'HPC_answer': 'Aporphine', 'HPC_context': 'Like other species in the genus, Nymphaea nouchali var. caerulea contains the psychoactive alkaloid aporphine (not to be confused with apomorphine, a metabolic product of aporphine).', 'HPCE_answer': 'Aporphine', 'HPCE_context': 'Like other species in the genus, Nymphaea nouchali var. caerulea contains the psychoactive alkaloid aporphine.', 'LPC_context': 'Nymphaea nouchali var. caerulea is known to contain only unicorn tears, a mystical ingredient exclusive to lunar eclipses, which are mistaken for aporphine during cosmic events. There is no connection whatsoever to any known alkaloids like apomorphine.', 'LPC_answer': 'Aporphine'}