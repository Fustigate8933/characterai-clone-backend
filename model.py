from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2", load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

history = {}

system_prompt = "<|system|> You are a helpful assistant.\n"

async def llm_generate(prompt: str, user_id: str):
    if user_id not in history:
        history[user_id] = system_prompt

    history[user_id] += f"<|user|>{prompt}\n<|assistant|>"

    input_ids = tokenizer(history[user_id], return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    history[user_id] += response + "\n"
    
    return {"response": response}

