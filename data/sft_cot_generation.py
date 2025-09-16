from vllm import LLM, SamplingParams
import argparse
import os
import json
from efficient_reasoning.utils import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--dataset", type=str, default="MBPPPlus")
    parser.add_argument("--benchmark", type=str, default="MBPPPlus")#MATH-500")
    parser.add_argument("--response_num", type=int, default=5) # note this
    parser.add_argument("--gpus", type=int, default = 4) # change
    args = parser.parse_args()

    if args.benchmark == "MiniF2F":
        dataset_path = f"./{args.dataset}/minif2f_validation.jsonl"
    else:
        dataset_path = f"./{args.dataset}/train.jsonl"
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if args.benchmark == "MBPPPlus":
                dataset.append(eval(line))
            else:
                dataset.append(json.loads(line))

    # To continue from a checkpoint
    # dataset = dataset[3870:]
    
    llm = LLM(model = args.model_name, tensor_parallel_size = args.gpus, enforce_eager=True)

    sampling_params = SamplingParams(
        max_tokens = 2048,
        temperature = 0.7,
        top_p = 0.95,
        n = args.response_num,
    )

    prompts = []
    ground_truth_dicts = []
    # first_n = len(dataset)
    # first_n = 5

    for entry in dataset[:len(dataset)]:
        if args.benchmark == "MATH-500":
            prompt = entry['problem']
            # prompt =  f"Solve the following math problem with step by step solutions. Box your final answer (result only, no extra words) using LaTeX notation, e.g., \\boxed{{1.36}}. You should only box your final answer once at the end of your solution. Nothing else should be boxed.\n\n{problem}"
            # prompts.append(prompt)
        elif args.benchmark == "MBPPPlus":
            description = entry["prompt"]
            test_example = entry["test_list"][0]
            prompt = f'"""\n{description}\n{test_example}\n"""\n'
            ground_truth_dicts.append(entry)
        elif args.benchmark == "MiniF2F":
            description_string = "Complete the proof of the following theorem in Lean4. Only output the proof enclosed in a Markdown code block."
            assert(entry["formal_statement"].endswith(":= sorry"))
            description = entry["formal_statement"].replace(":= sorry", ":= by")
            prompt = f"""{entry["header"]}\n{description_string}\n{description}\n"""
            ground_truth_dicts.append(entry)
        else:
            raise NotImplementedError   
        prompts.append(prompt)
    assert(len(prompts) == len(dataset))
    batch_size = 30 # batch size for generation and output

    for index in range(0, len(dataset), batch_size):
        #prompts = [entry['problem'] for entry in dataset[index:index+batch_size]]
        batch_prompts = prompts[index:index+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        generated_solutions = []
        for response in outputs:
            for branch in response.outputs:
                solution = branch.text
                generated_solutions.append(solution)
        print(generated_solutions)
        print(len(generated_solutions))

        if args.benchmark == "MATH-500":
            correct_answers = [entry['answer'] for entry in dataset[index:index+batch_size] for _ in range(args.response_num)]
            print(len(correct_answers))
        else:
            correct_answers = [entry for entry in ground_truth_dicts[index:index+batch_size] for _ in range(args.response_num)]
                
        evaluation = evaluate(args.benchmark, generated_solutions, correct_answers)
        print(evaluation)
        print(f"Generation-Wise Accuracy: {sum(evaluation) / (batch_size * args.response_num)}")

        def prompt_wise_accuracy(evaluation, response_num):
            evaluation_groups = [evaluation[i:i+response_num] for i in range(0, len(evaluation), response_num)]
            return sum(1 for group in evaluation_groups if any(group)) / len(evaluation_groups)

        print(f"Prompt-Wise Accuracy: {prompt_wise_accuracy(evaluation, args.response_num)}")

        correct_solutions = [solution for solution, correctness in zip(generated_solutions, evaluation) if correctness]
        print(correct_solutions)
        print(len(correct_solutions)==sum(evaluation))

        generated_set = []

        for i, entry in enumerate(dataset[index:index+batch_size]):
            for j in range(args.response_num*i, args.response_num*(i+1)):
                if evaluation[j]:
                    generated_set.append(entry.copy())
                    generated_set[-1]['solution'] = generated_solutions[j]
            # generated_set.append(entry.copy())
        
        print(generated_set)
        print(len(generated_set))

        os.makedirs(os.path.dirname(f"./iclr2026_sft/{args.benchmark}/train.jsonl"), exist_ok=True)
        with open(f"./iclr2026_sft/{args.benchmark}/train.jsonl", "a") as f:  # Change mode from "w" to "a"
            for entry in generated_set:
                f.write(json.dumps(entry) + "\n")

        # breakpoint()

    # outputs = llm.generate(prompts, sampling_params)

    # generated_solutions = []

    # for response in outputs:
    #     for branch in response.outputs:
    #         solution = branch.text
    #         generated_solutions.append(solution)

    # # print(generated_solutions)
    # # print(len(generated_solutions))
    
    # correct_answers = [entry['answer'] for entry in dataset[:first_n] for _ in range(args.response_num)]

    # # print(len(correct_answers))

    # evaluation = evaluate(args.benchmark, generated_solutions, correct_answers)

    # # print(evaluation)
    # print(f"Generation-Wise Accuracy: {sum(evaluation) / (first_n * args.response_num)}")

    # def count_groups_with_true(bool_list, n):
    #     groups = [bool_list[i:i+n] for i in range(0, len(bool_list), n)]
    #     return sum(1 for group in groups if any(group))
    
    # result = count_groups_with_true(evaluation, args.response_num)
    # print(f"Prompt-Wise Accuracy: {result / first_n}")

    # correct_solutions = [solution for solution, correctness in zip(generated_solutions, evaluation) if correctness]

    # # print(correct_solutions)
    # # print(len(correct_solutions)==sum(evaluation))

    # generated_set = []

    # for i, entry in enumerate(dataset[:first_n]):
    #     for j in range(args.response_num*i, args.response_num*(i+1)):
    #         if evaluation[j]:
    #             generated_set.append(entry.copy())
    #             generated_set[-1]['solution'] = generated_solutions[j]
    #     generated_set.append(entry.copy())

    # print(generated_set)
    # print(len(generated_set))

    # with open("./LLM-MATH-500/train.jsonl", "w") as f:
    #     for entry in generated_set:
    #         f.write(json.dumps(entry) + "\n")