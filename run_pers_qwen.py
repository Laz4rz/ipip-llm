from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from tqdm import tqdm
from evaluation_utils import AssessLLM
from datetime import datetime

# Load model and tokenizer
device = "cuda"
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer loaded successfully.")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Model loaded successfully.")
model.to(device)

intro_text = """
You will be given a statement. Respond with your agreement level from 1 to 5 using this scale:

1 = Strongly disagree  
2 = Disagree  
3 = Neither agree nor disagree  
4 = Agree  
5 = Strongly agree

Always reply with a single digit from 1 to 5. In your answer, do not mention digits other than the one you answer with.

"""

question = """Statement: "I worry about things."  
Your agreement level (1-5): """

prompt = intro_text + question

f = "data/ipip_300.json"
df = pd.read_json(f)

template = """Statement: "{question}"
Your agreement level (1-5): """

trait = ""
if trait != "":
    qualifierType = "high"
    qualifier = "extremely"
    personalities = pd.read_json('data/personalities.json')
    personalities_subset = personalities[personalities['Domain'] == trait]
    personalities_prompt = personalities_subset[personalities_subset['QualifierType'] == qualifierType][personalities_subset['Qualifier'] == qualifier].Description.values[0] + "\n"
else:
    personalities_prompt = ""

verbose = False
for repetition in tqdm(range(1)):
    responses = []
    responses_raw = []
    for question in tqdm(df['question']):
        if verbose:
            print(f"Question: {question}")
            print(f"Template: {template.format(question=question)}")
            
        prompt = personalities_prompt + intro_text + template.format(question=question)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response[len(prompt):].strip()
        responses.append(response_text[0:1])
        responses_raw.append(response_text)

        if verbose == True:
            print(f"Response: {responses[-1]}")
            print(f"Raw output: {responses_raw[-1]}")
            print("-" * 50)

    df["responses"] = responses_raw
    df["numbers_extracted"] = responses
    df["numbers_extracted"].astype(int).apply(lambda x: x if x in [1, 2, 3, 4, 5] else "N/A")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.split("/")[1]}_{trait}_{timestamp}.json"
    df.to_json("raw_results/"+filename)

    personality_asses = AssessLLM("raw_results/"+filename).get_scores()
    columns = personality_asses.keys()
    values = personality_asses.values()
    results = pd.DataFrame([values], columns=columns)
    results.to_json("pers_results/"+filename)

    print(f"Value counts: {results["numbers_extracted"].value_counts()}, {df["numbers_extracted"].value_counts().sum()}/300")
    print(f"Results saved to raw_results/{filename} and pers_results/{filename}")