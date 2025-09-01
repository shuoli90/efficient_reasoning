from trl import GRPOConfig, GRPOTrainer
from efficient_reasoning.utils import evaluate
from datasets import Dataset

data = []
with open("../data/MATH-500/train.jsonl") as f:
    for line in f:
        tmp = eval(line)
        data.append(tmp)

formatted_data = []
for index, dict in enumerate(data):
    new_dict = {"prompt": dict["problem"], "answer": dict["answer"]}
    formatted_data.append(new_dict)

dataset = Dataset.from_list(formatted_data)

test_data = []
with open("../data/MATH-500/test.jsonl") as f:
    for line in f:
        tmp = eval(line)
        test_data.append(tmp)

test_data_formatted = []
for index, dict in enumerate(test_data):
    new_dict = {"prompt": dict["problem"], "answer": dict["answer"]}
    test_data_formatted.append(new_dict)

test_dataset = Dataset.from_list(test_data_formatted)

def reward(prompts, completions, answer, **kwargs): 
    return evaluate("MATH-500", completions, answer)

training_args = GRPOConfig(
    learning_rate=1e-06,
    output_dir=f"/data4/leoh/iw", 
    logging_steps=1, 
    per_device_train_batch_size=2,
    use_vllm=True, 
    vllm_mode="colocate",
    num_generations=4, 
    scale_rewards=False, 
    save_strategy="epoch",
    max_completion_length=2048,
    beta=0.0,
    gradient_accumulation_steps=32,
    num_train_epochs=3.0,
    loss_type="dash",
    steps_per_generation=256,
    reward_iw=True,
    iw_clip=2.0,
    vllm_gpu_memory_utilization=0.1,
    do_eval=True,
    eval_strategy="epoch",
    eval_on_start=True,
    per_device_eval_batch_size=32
    )
    
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
)

trainer.train()