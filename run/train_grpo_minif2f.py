from trl.trainer import GRPOConfig, GRPOTrainer
from efficient_reasoning.utils import evaluate
from datasets import Dataset
import json
data = []

with open('../data/MiniF2F/minif2f_validation.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

formatted_data = []
for index, dict in enumerate(data):
    description_string = "Complete the proof of the following theorem in Lean4. Only output the proof enclosed in a Markdown code block."
    assert(dict["formal_statement"].endswith(":= sorry"))
    description = dict["formal_statement"].replace(":= sorry", ":= by")
    prompt = f"""{dict["header"]}\n{description_string}\n{description}\n"""
    new_dict = {"prompt": prompt, "answer": dict}
    formatted_data.append(new_dict)

dataset = Dataset.from_list(formatted_data)
print(formatted_data[0]["prompt"])
test_data = []

with open('../data/MiniF2F/minif2f_test.jsonl') as f:
    for line in f:
        test_data.append(json.loads(line))

test_data_formatted = []
for index, dict in enumerate(test_data):
    description_string = "Complete the proof of the following theorem in Lean4. Only output the proof enclosed in a Markdown code block."
    assert(dict["formal_statement"].endswith(":= sorry"))
    description = dict["formal_statement"].replace(":= sorry", ":= by")
    prompt = f"""{dict["header"]}\n{description_string}\n{description}\n"""
    new_dict = {"prompt": prompt, "answer": dict}
    test_data_formatted.append(new_dict)

test_dataset = Dataset.from_list(test_data_formatted)

def reward(prompts, completions, answer, **kwargs): 
    #return evaluate("MATH-500", completions, answer)
    return evaluate('MiniF2F', completions, answer)

#DASH A4 Config (for A8 change steps_per_generation to 32, for A4 set to 16)
# training_args = GRPOConfig(
#     max_prompt_length=1900,
#     learning_rate=1e-06,
#     output_dir=f"../results/minif2f_0.5B_dash_a4_grpo_loss_3",
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
#     steps_per_generation=16, #16, 32
#     reward_iw=True,
#     iw_clip=2.0,
#     vllm_gpu_memory_utilization=0.3,
#     do_eval=True,
#     eval_strategy="epoch",
#     eval_on_start=True,
#     per_device_eval_batch_size=16,
#     bf16=True,
# )

#GRPO Config
# training_args = GRPOConfig(
#     max_prompt_length=1900,
#     learning_rate=1e-06,
#     output_dir=f"../results/minif2f_0.5B_grpo_3",
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
#     output_dir=f"../results/minif2f_0.5B_dapo_3",
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
training_args = GRPOConfig(
    max_prompt_length=1900,
    learning_rate=1e-06,
    output_dir=f"../results/minif2f_0.5B_policy_gradient_3",
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
    # steps_per_generation=128,
    # reward_iw=True,
    # iw_clip=2.0,
    vllm_gpu_memory_utilization=0.3,
    do_eval=True,
    eval_strategy="epoch",
    eval_on_start=True,
    per_device_eval_batch_size=16,
    bf16=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
)

trainer.train()