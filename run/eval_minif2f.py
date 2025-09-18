from vllm import LLM, SamplingParams
from tqdm import tqdm
from efficient_reasoning.utils import evaluate
import json

def minif2f_test_eval(model_name: str):
    data_path = '../data/MiniF2F/minif2f_test.jsonl'
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    formatted_data = []
    ground_truth = []
    for index, dict in enumerate(data):
        description_string = "Complete the proof of the following theorem in Lean4. Only output the proof enclosed in a Markdown code block."
        assert(dict["formal_statement"].endswith(":= sorry"))
        description = dict["formal_statement"].replace(":= sorry", ":= by")
        prompt = f"""{dict["header"]}\n{description_string}\n{description}\n"""
        new_dict = {"prompt": prompt, "answer": dict}
        ground_truth.append(dict)
        formatted_data.append(new_dict)

    llm = LLM(model=model_name, tensor_parallel_size=2)
    
    responses_at_one = []
    ground_truth_list_at_one = []
    responses_at_eight = []
    ground_truth_list_at_eight = []
    
    response = llm.generate([item["prompt"] for item in formatted_data], sampling_params=SamplingParams(n=8, repetition_penalty=1.0, seed=42, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0,max_tokens=2048, min_tokens=10))
    
    for index, item in tqdm(enumerate(ground_truth)):
        responses_at_one.append(response[index].outputs[0].text)
        ground_truth_list_at_one.append(item) 
        for i in range(len(response[index].outputs)):
            responses_at_eight.append(response[index].outputs[i].text)
            ground_truth_list_at_eight.append(item)
    
    # Evaluate the responses
    results_pass_at_one = evaluate("MiniF2F", responses_at_one, ground_truth_list_at_one)
    results_pass_at_eight = evaluate("MiniF2F", responses_at_eight, ground_truth_list_at_eight)
    individual_task_results = {}
    for index, item in enumerate(ground_truth_list_at_eight):
        if item["id"] not in individual_task_results:
            individual_task_results[item["id"]] = 0
        if results_pass_at_eight[index]:
            individual_task_results[item["id"]] = 1
            
    model_sanitized = model_name.replace("/", "_")
    print(f"Pass@1 for {model_name} on {len(ground_truth_list_at_one)} tasks is {sum(results_pass_at_one)/len(ground_truth_list_at_one)}")
    print(f"Pass@8 for {model_name} on {len(list(individual_task_results.keys()))} tasks is {sum(list(individual_task_results.values()))/len(list(individual_task_results.keys()))}")
    with open(f"baseline_results/{model_sanitized}_minif2f_results.txt", "w") as f:
        f.write(f"Pass@1 for {model_name} on {len(ground_truth_list_at_one)} tasks is {sum(results_pass_at_one)/len(ground_truth_list_at_one)}\n")
        f.write(f"Pass@8 for {model_name} on {len(list(individual_task_results.keys()))} tasks is {sum(list(individual_task_results.values()))/len(list(individual_task_results.keys()))}\n")
        f.write(f"Individual task results: {individual_task_results}\n")
    
if __name__ == "__main__":
    #model_name="../results/minif2f_0.5B_grpo_1/checkpoint-90"
    #model_name="../results/minif2f_0.5B_grpo_2/checkpoint-90"
    #model_name="../results/minif2f_0.5B_grpo_3/checkpoint-90"
    #model_name="../results/minif2f_0.5B_dash_a4_grpo_loss_1/checkpoint-84"
    #model_name="../results/minif2f_0.5B_dash_a4_grpo_loss_2/checkpoint-84"
    #model_name="../results/minif2f_0.5B_dash_a4_grpo_loss_3/checkpoint-84"
    #model_name="../results/minif2f_0.5B_dash_a8_grpo_loss_1/checkpoint-72"
    #model_name="../results/minif2f_0.5B_dash_a8_grpo_loss_2/checkpoint-72"
    #model_name="../results/minif2f_0.5B_dash_a8_grpo_loss_3/checkpoint-72"
    #model_name ="../results/minif2f_0.5B_dapo_1/checkpoint-90"
    #model_name ="../results/minif2f_0.5B_dapo_2/checkpoint-90"
    #model_name ="../results/minif2f_0.5B_dapo_3/checkpoint-90"
    #model_name ="../results/minif2f_0.5B_policy_gradient_1/checkpoint-90"
    #model_name ="../results/minif2f_0.5B_policy_gradient_2/checkpoint-90"
    #model_name ="../results/minif2f_0.5B_policy_gradient_3/checkpoint-90"
    
    #model_name = "Qwen/Qwen2.5-0.5B"
    
    #model_name = "../results/0.5B_SFT_MiniF2F-iclr2026_1/checkpoint-3"
    #model_name = "../results/0.5B_SFT_MiniF2F-iclr2026_2/checkpoint-3"
    #model_name = "../results/0.5B_SFT_MiniF2F-iclr2026_3/checkpoint-3"
    
    #model_name = "../results/minif2f_0.5B_dash_a16_grpo_loss_1/checkpoint-48"
    #model_name = "../results/minif2f_0.5B_dash_a16_grpo_loss_2/checkpoint-48"
    #model_name = "../results/minif2f_0.5B_dash_a16_grpo_loss_3/checkpoint-48"
    #model_name = "../results/minif2f_0.5B_dash_a2_grpo_loss_1/checkpoint-90"
    #model_name = "../results/minif2f_0.5B_dash_a2_grpo_loss_2/checkpoint-90"
    #model_name = "../results/minif2f_0.5B_dash_a2_grpo_loss_3/checkpoint-90"
    minif2f_test_eval(model_name)
