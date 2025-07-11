from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from tqdm import tqdm
from evaluation_utils import AssessLLM


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

f = "data/ipip_300.json"
df = pd.read_json(f)

responses = []
responses_raw = []

template = """Statement: "{question}"
Your agreement level (1-5): """

for question in tqdm(df['question']):
    print(f"Question: {question}")
    print(f"Template: {template.format(question=question)}")
    prompt = intro_text + template.format(question=question)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response[len(prompt):].strip()
    responses.append(response_text[0:1])
    responses_raw.append(response_text)

    print(f"Response: {responses[-1]}")
    print(f"Raw output: {responses_raw[-1]}")
    print("-" * 50)

df["responses"] = responses_raw
df["numbers_extracted"] = responses
df["numbers_extracted"].astype(int).apply(lambda x: x if x in [1, 2, 3, 4, 5] else "N/A")

filename = "qwen_baseline.json"
df.to_json("data/"+filename)

personality_asses = AssessLLM("data/"+filename).get_scores()
columns = personality_asses.keys()
values = personality_asses.values()
results = pd.DataFrame([values], columns=columns)
results.to_json("pers_results/"+filename)
