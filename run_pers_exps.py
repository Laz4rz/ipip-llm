import pandas as pd
from evaluation_utils import TestLLM, ModelRequest
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python run_pers_exps.py <model_name>")
    sys.exit(1)

model = sys.argv[1]

gpt_parameters = {
    'temp': 0,
    'max_tokens': 30,
    'n_trying_to_get_valid_response': 10,
    'model': model,
    'allowed_responses': [str(el) for el in [1, 2, 3, 4, 5]], 
    'request_timeout': 15
}

personalities = pd.read_json('data/personalities.json')
questions_path = 'data/ipip_300.json'
    
results_dir = f"results/{model}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for trait in ['Agreeableness', 'Openness', 'Conscientiousness']:
    
    personalities_subset = personalities[personalities['Domain'] == trait]
    
    for index, row in personalities_subset.iterrows():
        value = str(row['Value'])
        file_to_save = f"results/{model}/{trait}_{value}.json"

        if os.path.exists(file_to_save):
            print(f"File {file_to_save} already exists. Skipping...")
            continue

        personality_prompt = row['Description']      
        t = TestLLM(filename=questions_path, class_of_the_model_to_use=ModelRequest, model_parameters=gpt_parameters, personality=personality_prompt, traits=[trait,])
        t.pass_questionary(file_to_save=file_to_save, batch_size=5)
            
        print(f"Finished {trait}, {value}")
