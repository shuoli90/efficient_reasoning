from efficient_reasoning.utils import evaluate, Benchmark
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
from tqdm import tqdm
import math
import os
from datetime import datetime

if __name__ == "__main__":

    current_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--benchmark', type=str, default='MATH-500', choices=['MATH-500', 'AIME', 'BigCodeBench'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_token', type=int, default=2048)
    args = parser.parse_args()
    
    if args.benchmark == 'MATH-500':
        data_path = '../data/MATH-500/test.jsonl'
    elif args.benchmark == 'AIME':
        data_path = '../data/AIME_2024/AIME_2024_I&II_ArtOfProblemSolving.jsonl'
    else:
        data_path = '/home/lishuo1/efficient_reasoning/data/BigCodeBench/test.jsonl'
    
    data = []
    with open(data_path) as f:
        for line in f:
            tmp = eval(line)
            data.append(tmp)
    
    sampling_params = SamplingParams(
        max_tokens=args.max_token,
        temperature=0.7,
        top_p=0.95,
        n=1,
    )
    
    # load the model
    llm = LLM(model=args.model_name, gpu_memory_utilization=0.6)
    
    correct = 0
    for i in tqdm(range(0, len(data), args.batch_size), total=math.ceil(len(data) / args.batch_size)):
        batch = data[i:i+args.batch_size]
        targets = [question['answer'] for question in batch]
        problems = [question['problem'] for question in batch]
        # prompts = [Template.format(problem=problem) for problem in problems]
        outputs = llm.generate(
            problems,
            sampling_params=sampling_params,
        )
        responses = [output.outputs[0].text for output in outputs]
        # evaluate the responses
        rewards = evaluate(args.benchmark, responses, targets)
        correct += sum(rewards)
    print(f"Accuracy: {correct / len(data)}")

    if not os.path.exists('results'):
        os.makedirs('results')
    
    with open(f"results/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.txt", 'w') as f:
        f.write(f"Model Path: {model_path}\n") # if base model, should output base mode name
        f.write(f"Accuracy: {correct / len(data)}\n")
        f.write(f"Representative LLM Response [0]: {responses[0]}\n")
        f.write(f"Representative LLM Response [10]: {responses[50]}\n")
        f.write(f"Representative LLM Response [101]: {responses[100]}\n\n")
    