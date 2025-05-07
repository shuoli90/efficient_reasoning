import re
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from efficient_reasoning.grpo_trainer import GRPOTrainer
from efficient_reasoning.grpo_config import GRPOConfig
import wandb
import torch.distributed as dist
from efficient_reasoning.utils import evaluate
#from latex2sympy2_extended import NormalizationConfig
#from math_verify import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig, parse, verify

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values are: 'accuracy', 'format'
    """

    reward_funcs: list[str] = field(
        # default_factory=lambda: ['accuracy', 'format'],
        # default_factory=lambda: ['accuracy'],
        default_factory=lambda: ['bigcodebench_accuracy'],
        metadata={
            # "help": "List of reward functions. Possible values are: 'accuracy', 'format'"
            "help": "List of reward functions. Possible values are: 'accuracy'"
        }
    )
    
def accuracy_reward(completions, solution, **kwargs):
    result = evaluate('MATH-500', completions, solution)
    result = [1 if r else 0 for r in result]
    return result

def bigcodebench_accuracy_reward(completions, solution, **kwargs):
    result = evaluate('BigCodeBench', completions, solution)
    result = [1 if r else 0 for r in result]
    return result

# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is the same as the ground truth."""
#     # contents = [completion[0]["content"] for completion in completions]
#     contents = completions
#     rewards = []
#     for content, sol in zip(contents, solution):
#         gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[
#             LatexExtractionConfig(), 
#             ExprExtractionConfig(), 
#             StringExtractionConfig()])
#         if len(gold_parsed) != 0:
#             # We require the answer to be provided in correct latex (no malformed operators)
#             answer_parsed = parse(
#                 content,
#                 extraction_config=[
#                     LatexExtractionConfig(
#                         normalization_config=NormalizationConfig(
#                             nits=False,
#                             malformed_operators=False,
#                             basic_latex=True,
#                             equations=True,
#                             boxed=True,
#                             units=True,
#                         ),
#                         # Ensures that boxed is tried first
#                         boxed_match_priority=0,
#                         try_extract_without_anchor=False,
#                     )
#                 ],
#                 extraction_mode="first_match",
#             )
#             # Reward 1 if the content is the same as the ground truth, 0 otherwise
#             try:
#                 reward = float(verify(answer_parsed, gold_parsed))
#             except Exception as e:
#                 print(e)
#                 reward = 0.0
#         else:
#             # If the gold solution is not parseable, we reward 1 to skip this example
#             reward = 1.0
#             print("Failed to parse gold solution: ", sol)
#         rewards.append(reward)
#     return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "bigcodebench_accuracy": bigcodebench_accuracy_reward,
    # "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# LATEX_TEMPLATE = "$\\boxed{sol}$"

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    #data_path = '../data/MATH-500/train.jsonl'
    #Modified for BigCodeBench
    data_path = '../data/BigCodeBench/train.jsonl'
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(eval(line))
    
    formatted_data = []
    for index, item in enumerate(data):
        #new_dict = {"prompt": item["problem"], "solution": item["answer"]}
        #Modified for BigCodeBench
        new_dict = {"prompt": item["problem"], "solution": item["solution"]}
        formatted_data.append(new_dict)
    formatted_data = [formatted_data[0]]
    dataset = Dataset.from_list(formatted_data)

    # Format into conversation
    # def make_conversation(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": SYSTEM_PROMPT},
    #             {"role": "user", "content": example["problem"]},
    #         ],
    #         "solution": LATEX_TEMPLATE.format(sol=example["answer"]),
    #     }

    # dataset = dataset.map(make_conversation)
    # dataset = dataset.remove_columns(["problem", "answer", "level", "subject", "unique_id"])

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":

    # Initialize distributed processing
    dist.init_process_group(backend="nccl")  # Use 'gloo' for CPU

    # # Check the process rank
    rank = dist.get_rank()
    is_main_process = rank == 0 

    if is_main_process:
        wandb.init(
            # set the wandb project where this run will be logged
            project="llmrl",
        )

    vllm_server_configs = [
        {"host": "0.0.0.0", "server_port": 8003, 'group_port': 51222},
        # {"host": "0.0.0.0", "server_port": 8004, 'group_port': 51223},
        # {"host": "158.130.55.13", "server_port": 8002},
        # {"host": "158.130.55.13", "server_port": 8003},
    ]

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.vllm_server_configs = vllm_server_configs
    main(script_args, training_args, model_args)
