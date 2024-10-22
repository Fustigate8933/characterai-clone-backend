from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer (adjust to your specific model)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

