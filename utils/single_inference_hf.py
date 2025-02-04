from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input text for inference
input_text = "Fix the grammar of the given sentences. Who is the president of the United States? An ice cream have been elected as the president of the United States in 2024."

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output from the model
outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)