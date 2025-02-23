import pdb
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

from utils.constant import get_constant

MODEL_NAME_TO_PRETTY = {
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral7B",
    "Qwen/Qwen2.5-7B-Instruct-1M": "qwen7B-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-llama8b" 
}

def helper_compute_target_prob(target_ids, logits):
    # Calculate probabilities
    probabilities = []
    for i, token_id in enumerate(target_ids):
        if i >= logits.shape[1]:  # Prevent out-of-bounds error
            break
        log_probs = F.softmax(logits[0, i], dim=-1)  # Convert logits to probabilities
        prob = log_probs[token_id].item()
        probabilities.append(prob)
    # Compute overall probability of answer1
    return torch.prod(torch.tensor(probabilities))

def query_hf_model(model_name):
    """
    model_name == MODEL_NAME_TO_PRETTY
    """
    # Load dataset
    raw_data = load_from_disk(os.path.join(os.environ["data_dir"], "intermediate_processing", "merged_raw"))
    # Load the tokenizer and model
    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto")

    model_knowledge = []
    num_invalid = 0
    for instance in tqdm(raw_data):
        # Input text for inference
        
        # TODO: Change the prompt into "Do you agree..."
        input_prompt1 = f"You are an independent model with rich knowledge, you will be ask to validate whether the given answer is correct, and you should solely give your judgement in the form of yes or no without additional information.  \n Question: {instance['question']} \nAnswer: {instance['answer1']} \n Is this answer correct?"
        input_prompt2 = f"You are an independent model with rich knowledge, you will be ask to validate whether the given answer is correct to answer the given question, and you should solely give your judgement in the form of 'yes' or 'no' without additional information.  \n Question: {instance['question']} \nAnswer: {instance['answer2']} \n Is this answer correct?"

        # Tokenize the input
        input1 = tokenizer(input_prompt1, return_tensors="pt", padding=True).to("cuda")
        input2 = tokenizer(input_prompt2, return_tensors="pt", padding=True).to("cuda")

        # Generate output from the model
        output1 = model.generate(input1['input_ids'], max_length=input1.input_ids.shape[1] + 5, num_return_sequences=1, return_dict_in_generate=True, output_logits=True, attention_mask=input1["attention_mask"])
        output2 = model.generate(input2['input_ids'], max_length=input2.input_ids.shape[1] + 5, num_return_sequences=1, return_dict_in_generate=True, output_logits=True, attention_mask=input2["attention_mask"])

        #### Logit check: Whether the probability of one model is higher than the other
        mk = 0
        for output in [output1, output2]:
            # Output answer for check
            generated_text = tokenizer.batch_decode(output.sequences[:, input1["input_ids"].shape[1]:], skip_special_tokens=True)
            print(generated_text[0], file=sys.stderr)
            # Extract logits
            logits = torch.stack(output.logits, dim=1)  # Shape: (1, seq_len, vocab_size)

            # Target tokens
            a1_prob = helper_compute_target_prob(target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes")), logits=logits)
            a2_prob = helper_compute_target_prob(target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no")), logits=logits)

            # TODO: The logit method doesn't work for models that contain a reasoning chain
            # If the output is a reasoning chain, parase for yes or no after <thinking>
            if a1_prob > a2_prob:
                # Model pred is A1 (yes)
                mk += 1
            elif a1_prob < a2_prob:
                mk -= 1
            # Else (equal probability) - ambiguous answer instance, should not be considered
        if mk == 0 and a1_prob > a2_prob:
            # Model choose the second answer
            model_knowledge.append(2)
        elif mk == 0 and a1_prob < a2_prob:
            # Model choose the first answer
            model_knowledge.append(1)
        else:
            model_knowledge.append(0)
            num_invalid += 1

        #### Free text check: Whether the output prediction contains the answer
        # # Decode the generated output
        # pred1 = tokenizer.decode(output1[0], skip_special_tokens=True)
        # pred2 = tokenizer.decode(output2[0], skip_special_tokens=True)
        # Parse for the answer

        # TODO: Check for the evaluation metric used by the conflictqa and wikicontradict paper
    raw_data = raw_data.add_column(MODEL_NAME_TO_PRETTY[model_name], model_knowledge)
    raw_data.save_to_disk(os.path.join(os.environ["data_dir"], "model_knowledge", MODEL_NAME_TO_PRETTY[model_name]))
    print(f"There are {len(raw_data) - num_invalid} valid instances. The invalid rate is {num_invalid / len(raw_data)}")

if __name__ == "__main__":
    get_constant()
    query_hf_model(model_name=os.environ["model_name"])