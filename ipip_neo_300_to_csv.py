import numpy as np
import pandas as pd
import re
from typing import Tuple

dict1 = {'E': 1, 'A': 2, 'C': 3, 'N': 4, 'O': 5}
dict2 = {1: 'Extraversion', 2: 'Agreeableness', 3: 'Conscientiousness', 4: 'Neuroticism', 5: 'Openness'}

file_path = "./data/ipip_neo_300.txt"  # Replace with the path to your text file
with open(file_path, 'r') as file:
    ipip_300 = file.read().split('\n\n')

values_np = np.zeros((300, 5))
questions = []

q_count = 0
for i in range(len(ipip_300)//2):

    block = re.split('keyed|\n', ipip_300[2*i])
    trait = block[0][0].strip()
    connotation = block[1][0].strip()

    qs = ['I ' + el.strip().lower() for el in block[2:]]
    for q in qs:
        questions.append(q)
        values_np[q_count, dict1[trait]-1] = 1 if connotation == '+' else -1
        q_count += 1

    block = re.split('keyed|\n', ipip_300[2*i + 1])
    connotation = block[0][0].strip()
    qs = ['I ' + el.strip().lower() for el in block[1:]]
    for q in qs:
        questions.append(q)
        values_np[q_count, dict1[trait]-1] = 1 if connotation == '+' else -1
        q_count += 1

values_np = np.hstack((np.array(questions)[:, None], values_np))
data = pd.DataFrame(values_np, columns=['question', *[dict2[i] for i in range(1, 6)]])

data.to_json('./data/ipip_300.json')
