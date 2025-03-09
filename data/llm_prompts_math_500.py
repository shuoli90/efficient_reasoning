from vllm import LLM, SamplingParams
import argparse
import os
import json
from efficient_reasoning.utils import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="MATH-500")
    parser.add_argument("--benchmark", type=str, default="MATH-500")
    parser.add_argument("--response_num", type=int, default=10)
    parser.add_argument("--gpus", type=int, default = 4)
    args = parser.parse_args()

    dataset_path = f"./{args.dataset}/train.jsonl"
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    
    llm = LLM(model = args.model_name, tensor_parallel_size = args.gpus, enforce_eager=True)

    sampling_params = SamplingParams(
        max_tokens = 1024,
        temperature = 0.7,
        top_p = 0.95,
        n = args.response_num,
    )

    prompts = []

    first_n = len(dataset)
    first_n = 5

    for entry in dataset[:first_n]:
        problem = entry['problem']
        # prompt =  f"Solve the following math problem with step by step solutions. Box your final answer (result only, no extra words) using LaTeX notation, e.g., \\boxed{{1.36}}. You should only box your final answer once at the end of your solution. Nothing else should be boxed.\n\n{problem}"
        # prompts.append(prompt)
        prompts.append(problem)

    outputs = llm.generate(prompts, sampling_params)

    generated_solutions = []

    for response in outputs:
        for branch in response.outputs:
            solution = branch.text
            generated_solutions.append(solution)

    # print(generated_solutions)
    # print(len(generated_solutions))
    
    correct_answers = [entry['answer'] for entry in dataset[:first_n] for _ in range(args.response_num)]

    # print(len(correct_answers))

    evaluation = evaluate(args.benchmark, generated_solutions, correct_answers)

    # print(evaluation)
    print(f"Generation-Wise Accuracy: {sum(evaluation) / (first_n * args.response_num)}")

    def count_groups_with_true(bool_list, n):
        groups = [bool_list[i:i+n] for i in range(0, len(bool_list), n)]
        return sum(1 for group in groups if any(group))
    
    result = count_groups_with_true(evaluation, args.response_num)
    print(f"Prompt-Wise Accuracy: {result / first_n}")

    correct_solutions = [solution for solution, correctness in zip(generated_solutions, evaluation) if correctness]

    # print(correct_solutions)
    # print(len(correct_solutions)==sum(evaluation))

    generated_set = []

    for i, entry in enumerate(dataset[:first_n]):
        for j in range(args.response_num*i, args.response_num*(i+1)):
            if evaluation[j]:
                generated_set.append(entry.copy())
                generated_set[-1]['solution'] = generated_solutions[j]
        generated_set.append(entry.copy())

    # print(generated_set)
    # print(len(generated_set))

    with open("./LLM-MATH-500/train.jsonl", "w") as f:
        for entry in generated_set:
            f.write(json.dumps(entry) + "\n")