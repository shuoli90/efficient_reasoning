from vllm import LLM, SamplingParams
from tqdm import tqdm
from efficient_reasoning.utils import evaluate

def mbppplus_test_eval(model_name: str):
    data_path = '../data/MBPPPlus/test.jsonl'
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(eval(line))


    formatted_data = []
    for index, item in enumerate(data):
        #new_dict = {"prompt": item["problem"], "solution": item["answer"]}
        #Modified for BigCodeBench
        description = item["prompt"]
        test_example = item["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        new_dict = {"prompt": prompt, "solution": item}
        formatted_data.append(new_dict)

    llm = LLM(model=model_name, tensor_parallel_size=2)
    
    responses_at_one = []
    ground_truth_list_at_one = []
    responses_at_eight = []
    ground_truth_list_at_eight = []
    
    response = llm.generate([item["prompt"] for item in formatted_data], sampling_params=SamplingParams(n=8, repetition_penalty=1.0, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0,max_tokens=2048, min_tokens=10))
    for index, item in tqdm(enumerate(formatted_data)):
        responses_at_one.append(response[index].outputs[0].text)
        ground_truth_list_at_one.append(item["solution"]) 
        for i in range(len(response[index].outputs)):
            responses_at_eight.append(response[index].outputs[i].text)
            ground_truth_list_at_eight.append(item["solution"])
    
    # Evaluate the responses
    results_pass_at_one = evaluate("MBPPPlus", responses_at_one, ground_truth_list_at_one)
    results_pass_at_eight = evaluate("MBPPPlus", responses_at_eight, ground_truth_list_at_eight)
    individual_task_results = {}
    for index, item in enumerate(ground_truth_list_at_eight):
        if item["task_id"] not in individual_task_results:
            individual_task_results[item["task_id"]] = 0
        if results_pass_at_eight[index]:
            individual_task_results[item["task_id"]] = 1
    model_sanitized = model_name.replace("/", "_")
    print(f"Pass@1 for {model_name} on {len(ground_truth_list_at_one)} tasks is {sum(results_pass_at_one)/len(ground_truth_list_at_one)}")
    print(f"Pass@8 for {model_name} on {len(list(individual_task_results.keys()))} tasks is {sum(list(individual_task_results.values()))/len(list(individual_task_results.keys()))}")
    with open(f"baseline_results/{model_sanitized}_mbppplus_results.txt", "w") as f:
        f.write(f"Pass@1 for {model_name} on {len(ground_truth_list_at_one)} tasks is {sum(results_pass_at_one)/len(ground_truth_list_at_one)}\n")
        f.write(f"Pass@8 for {model_name} on {len(list(individual_task_results.keys()))} tasks is {sum(list(individual_task_results.values()))/len(list(individual_task_results.keys()))}\n")
        f.write(f"Individual task results: {individual_task_results}\n")
    
if __name__ == "__main__":
    #model_name = "Qwen/Qwen2.5-Math-1.5B"
    model_name="../scripts/results/1_grpo"
    mbppplus_test_eval(model_name)
