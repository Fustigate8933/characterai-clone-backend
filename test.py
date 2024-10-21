from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = [
    {"role": "system", "content": "You are a helpful AI assistant for travel tips and recommendations."},
]

def format_history(messages):
    formatted = ""
    for message in messages:
        if message["role"] == "system":
            formatted += f"System: {message['content']}\n"
        elif message["role"] == "user":
            formatted += f"Human: {message['content']}\n"
        else:  # assistant
            formatted += f"AI: {message['content']}\n"
    return formatted + "AI: "


def generate_response(history):
    formatted = format_history(history)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        no_repeat_ngram_size=2,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("AI: ")[-1].strip()



while True:
    x = input("Input message: ")
    conversation_history.append({"role": "user", "content": x})
    assistant_response = generate_response(conversation_history)
    conversation_history.append({"role": "assistant", "content": assistant_response})
    print(f"Assistant: {assistant_response}")

