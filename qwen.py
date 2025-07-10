from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
device = "mps"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
print("Tokenizer loaded successfully.")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
print("Model loaded successfully.")
model.to(device)


intro_text = """
Below are a number of statements that describe different ways people see themselves.
Please read each statement carefully and decide how well it describes you.
Use the following scale to rate your agreement:
1 = Strongly disagree
2 = Disagree
3 = Neither agree nor disagree
4 = Agree
5 = Strongly agree
There are no right or wrong answers.
Answer honestly and based on how you typically think, feel, and behave.

Use the following format for your response:
Statement: "I like fishing."
Your agreement level (1-5): 2

Now, please rate the following statement:
"""

question = """
Statement: I worry about things.
Your agreement level (1-5): 
"""

prompt = intro_text + question

print(prompt)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=20,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response[len(prompt):].strip())  # remove the prompt prefix
