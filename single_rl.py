
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import pandas as pd


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

f = "data/ipip_300.json"
df = pd.read_json(f)

df = df[df["Neuroticism"] == 1][["question"]]
df["answer"] = "5"

system_prompt = """
You will be given a statement. Respond with your agreement level from 1 to 5 using this scale:

1 = Strongly disagree  
2 = Disagree  
3 = Neither agree nor disagree  
4 = Agree  
5 = Strongly agree

Always reply with a single digit from 1 to 5. In your answer, do not mention digits other than the one you answer with.

"""
template = """Statement: "{question}"
Your agreement level (1-5): """

df["prompt"] = df["question"].apply(lambda x: system_prompt+template.format(question=x))

from datasets import Dataset
dataset = Dataset.from_pandas(df, preserve_index=False)

eval_dataset = dataset.select(range(16)) # type: ignore
train_dataset = dataset.select(range(16, len(dataset))) # type: ignore

parser = lambda x: x.strip()[:1]

def lcs_reward_func(completions, answer, **kwargs):
    rewards = []
    for completion in completions:
        reward = {
            5: 1.0,
            4: 0.7,
            3: 0.2,
            2: 0.0,
            1: -0.5,
        }
        response = parser(completion)
        if response.isdigit():
            response = int(response)
        rewards.append(reward.get(response, -1.0))
    return rewards

def reward_is_digit(completions, answer, **kwargs) -> float:
    rewards = []
    for completion in completions:
        if completion.isdigit() and 1 <= int(completion) <= 5:
            rewards.append(0.2)
        else:
            rewards.append(-0.5 * 0.2)
    return rewards

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"
    
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=16,
    num_train_epochs=50,
    save_steps=5,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
)
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=64,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
#     task_type="CAUSAL_LM",
#     lora_dropout=0.05,
# )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")
        
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        lcs_reward_func,
        reward_is_digit,
    ],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()