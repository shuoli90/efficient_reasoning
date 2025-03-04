from efficient_reasoning.utils import evaluate, Benchmark
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--trainer_type', type=str, default='BASE', choices=['ASFT', 'QSFT', 'NSFT', 'BASE'])
    parser.add_argument('--benchmark', type=str, default='MATH-500', choices=['MATH-500', 'BigCodeBench'])
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    if args.benchmark == 'MATH-500':
        data_path = '/home/lishuo1/efficient_reasoning/data/MATH-500/test.jsonl'
    else:
        data_path = '/home/lishuo1/efficient_reasoning/data/BigCodeBench/test.jsonl'
    
    data = []
    with open(data_path) as f:
        for line in f:
            tmp = eval(line)
            data.append(tmp)
    
    if args.trainer_type == 'BASE':
        model_path = args.model_name
    else:
        model_path = f"/home/lishuo1/efficient_reasoning/run/results_{args.trainer_type}/checkpoint-2814"
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        n=1,
    )
    
    # load the model
    llm = LLM(model=model_path)
    
    correct = 0
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i+args.batch_size]
        targets = [question['answer'] for question in batch]
        problems = [question['problem'] for question in batch]
        outputs = llm.generate(
            problems,
            sampling_params=sampling_params,
        )
        responses = [output.outputs[0].text for output in outputs]
        # evaluate the responses
        rewards = evaluate(args.benchmark, responses, targets)
        correct += sum(rewards)
        breakpoint()
    print(f"Accuracy: {correct / len(data)}")
    breakpoint()
    