from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
print("Tokenizer loaded successfully.")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
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

Statement: "I like drones."
Your agreement level (1-5): 5

Statement: "I like painting."
Your agreement level (1-5): 1

Statement: "I jump."
Your agreement level (1-5): 3

"""

question = """
Statement: I worry about things.
Your agreement level (1-5): 
"""

prompt = intro_text + question

prompt2 = """
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

Statement: "I worry about things."  
Your agreement level (1-5): """

print(prompt2)

# Tokenize
inputs = tokenizer(prompt2, return_tensors="pt").to(device)

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
response_text = response
print(f"Response: '''\n{response_text}\n'''")
