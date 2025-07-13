from datasets import load_dataset
import verifiers as vf
import pandas as pd

#model = 'Qwen/Qwen2.5-1.5B-Instruct'
"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/reverse_text.py
"""


model_name = 'Qwen/Qwen2.5-0.5B'
print(f'Using model {model_name}')

f = "data/ipip_300.json"
df = pd.read_json(f)

df = df[df["Neuroticism"] == 1][["question"]]
df["answer"] = "5"

from datasets import Dataset
hf_dataset = Dataset.from_pandas(df, preserve_index=False)

eval_dataset = dataset.select(range(16)) # type: ignore
train_dataset = dataset.select(range(16, len(dataset))) # type: ignore
breakpoint()

parser = lambda x: x.strip()[:1]
system_prompt = f"""Reverse the given text.

Respond in the following format:
{parser.get_format_str()}"""

def lcs_reward_func(completion, answer, **kwargs) -> float:
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
    return reward.get(response, -1.0)

def reward_is_digit(completion, answer, **kwargs) -> float:
    if completion.isdigit() and 1 <= int(completion) <= 5:
        return 1
    else:
        return -0.5

rubric = vf.Rubric(funcs=[
	lcs_reward_func,
	reward_is_digit,
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=train_dataset, # type: ignore
    eval_dataset=eval_dataset, # type: ignore
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)
args = vf.grpo_defaults(run_name='reverse_text_warmup')
args.num_iterations = 2
args.per_device_train_batch_size = 12
args.num_generations = 12
args.gradient_accumulation_steps = 8
args.eval_strategy = "steps"
args.eval_steps = 10
args.max_steps = 100

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    #peft_config=vf.lora_defaults(),
    args=args
)
trainer.train()