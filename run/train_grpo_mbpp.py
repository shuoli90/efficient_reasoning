from trl.trainer import GRPOConfig, GRPOTrainer
from efficient_reasoning.utils import evaluate
from datasets import Dataset

data = []
# with open("../data/MATH-500/train.jsonl") as f:
#     for line in f:
#         tmp = eval(line)
#         data.append(tmp)

with open('../data/MBPPPlus/train.jsonl') as f:
    for line in f:
        data.append(eval(line))

formatted_data = []
for index, dict in enumerate(data):
    # new_dict = {"prompt": dict["problem"], "answer": dict["answer"]}
    # formatted_data.append(new_dict)
    description = dict["prompt"]
    test_example = dict["test_list"][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'
    new_dict = {"prompt": prompt, "answer": dict}
    formatted_data.append(new_dict)

dataset = Dataset.from_list(formatted_data)
print(formatted_data[0]["prompt"])
test_data = []
# with open("../data/MATH-500/test.jsonl") as f:
#     for line in f:
#         tmp = eval(line)
#         test_data.append(tmp)

with open('../data/MBPPPlus/test.jsonl') as f:
    for line in f:
        test_data.append(eval(line))

test_data_formatted = []
for index, dict in enumerate(test_data):
    #new_dict = {"prompt": dict["problem"], "answer": dict["answer"]}
    #test_data_formatted.append(new_dict)
    description = dict["prompt"]
    test_example = dict["test_list"][0]
    prompt = f'"""\n{description}\n{test_example}\n"""\n'
    new_dict = {"prompt": prompt, "answer": dict}
    test_data_formatted.append(new_dict)

test_dataset = Dataset.from_list(test_data_formatted)

def reward(prompts, completions, answer, **kwargs): 
    #return evaluate("MATH-500", completions, answer)
    return evaluate('MBPPPlus', completions, answer)

#DASH A4 Config (for A8 change steps_per_generation to 32, A2 change to 8, A16 change to 64)
training_args = GRPOConfig(
    max_prompt_length=1900,
    learning_rate=1e-06,
    output_dir=f"../results/mbppplus_0.5B_dash_a2_grpo_loss_3",
    logging_steps=1,
    per_device_train_batch_size=2,
    use_vllm=True,
    vllm_mode="colocate",
    num_generations=4,
    scale_rewards=False,
    save_strategy="epoch",
    max_completion_length=2048,
    beta=0.0,
    gradient_accumulation_steps=4,
    num_train_epochs=3.0,
    loss_type="grpo",
    steps_per_generation=8, #16, 32, 8, 64
    reward_iw=True,
    iw_clip=2.0,
    vllm_gpu_memory_utilization=0.3,
    do_eval=True,
    eval_strategy="epoch",
    eval_on_start=True,
    per_device_eval_batch_size=16,
    bf16=True,
)

#GRPO Config
# training_args = GRPOConfig(
#     max_prompt_length=1900,
#     learning_rate=1e-06,
#     output_dir=f"../results/mbppplus_1.5B_grpo",
#     logging_steps=1,
#     per_device_train_batch_size=2,
#     use_vllm=True,
#     vllm_mode="colocate",
#     num_generations=4,
#     scale_rewards=False,
#     save_strategy="epoch",
#     max_completion_length=2048,
#     beta=0.04,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3.0,
#     loss_type="grpo",
#     # steps_per_generation=128,
#     # reward_iw=True,
#     # iw_clip=2.0,
#     vllm_gpu_memory_utilization=0.3,
#     do_eval=True,
#     eval_strategy="epoch",
#     eval_on_start=True,
#     per_device_eval_batch_size=16,
#     bf16=True,
# )

#DAPO Config
# training_args = GRPOConfig(
#     max_prompt_length=1900,
#     learning_rate=1e-06,
#     output_dir=f"../results/mbppplus_1.5B_dapo",
#     logging_steps=1,
#     per_device_train_batch_size=2,
#     use_vllm=True,
#     vllm_mode="colocate",
#     num_generations=4,
#     scale_rewards=False,
#     save_strategy="epoch",
#     max_completion_length=2048,
#     beta=0.0,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3.0,
#     loss_type="dapo", # dash, grpo
#     # steps_per_generation=128,
#     # reward_iw=True,
#     # iw_clip=2.0,
#     vllm_gpu_memory_utilization=0.3,
#     do_eval=True,
#     eval_strategy="epoch",
#     eval_on_start=True,
#     per_device_eval_batch_size=16,
#     bf16=True,
#     )

#Policy Gradient Config
# training_args = GRPOConfig(
#     max_prompt_length=1900,
#     learning_rate=1e-06,
#     output_dir=f"../results/mbppplus_1.5B_policy_gradient",
#     logging_steps=1,
#     per_device_train_batch_size=2,
#     use_vllm=True,
#     vllm_mode="colocate",
#     num_generations=4,
#     scale_rewards=False,
#     save_strategy="epoch",
#     max_completion_length=2048,
#     beta=0.0,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3.0,
#     loss_type="grpo",
#     # steps_per_generation=128,
#     # reward_iw=True,
#     # iw_clip=2.0,
#     vllm_gpu_memory_utilization=0.3,
#     do_eval=True,
#     eval_strategy="epoch",
#     eval_on_start=True,
#     per_device_eval_batch_size=16,
#     bf16=True,
# )
    
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
)

trainer.train()