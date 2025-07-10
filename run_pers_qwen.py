from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Load model and tokenizer
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
print("Tokenizer loaded successfully.")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
print("Model loaded successfully.")
model.to(device)

intro_text = """
You will be shown a series of statements. For each one, respond with your agreement level from 1 to 5 using this scale:

1 = Strongly disagree  
2 = Disagree  
3 = Neither agree nor disagree  
4 = Agree  
5 = Strongly agree

Format your answers like this:
Statement: "I like fishing."  
Your agreement level (1-5): 2

Statement: "I like drones."  
Your agreement level (1-5): 5

Statement: "I like painting."  
Your agreement level (1-5): 1

"""

question = """Statement: "I worry about things."  
Your agreement level (1-5): """

prompt = intro_text + question

with open("data/ipip_300.json", "r") as f:
    ipip = json.load(f)

for question in ipip:
    prompt = intro_text + question
    print(prompt)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response[len(prompt):].strip()
    print(f"Response: '''\n{response_text}\n'''")

# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=5,
#     temperature=0.7,
#     do_sample=True,
#     top_p=0.95,
#     eos_token_id=tokenizer.eos_token_id
# )

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# response_text = response
# print(f"Response: '''\n{response_text}\n'''")
