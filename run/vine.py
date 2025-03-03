import os
from transformers import AutoTokenizer
import json
import argparse
from efficient_reasoning.games import Vine
from vllm import LLM, SamplingParams
from tqdm import tqdm

def save_games(games, model_name, dataset):
    if not os.path.exists(f'../collected/{model_name}_{dataset}'):
        os.makedirs(f'../collected/{model_name}_{dataset}')
        
    with open(f'../collected/{model_name}_{dataset}/{model_name}_{dataset}.jsonl', 'w') as f:
        for game in games:
            json.dump(game, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="MATH-500")
    args = parser.parse_args()  
    
    path = f'../data/{args.dataset}/train.jsonl'
    datapoints = []
    with open(path, 'r') as f:
        for line in f:
            datapoints.append(json.loads(line))
   
    #datapoints = datapoints[206:207]
    end_of_text_token = "<|end_of_text|>"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(model=args.model, 
            tensor_parallel_size=4)
    
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        n=8,
    )
    
    model_name = args.model.split("/")[-1]
    step_limit = 10
    from collections import defaultdict
    
    #benchmark = "MATH-500"
    benchmark = "BigCodeBench"
    games = []
    for i, problem in tqdm(enumerate(datapoints), total=len(datapoints)):
        if benchmark == "BigCodeBench":
            demonstration_steps = [problem['problem']] + problem["solution"].strip().split("\n")[:-1]
            demonstration_steps = [x for x in demonstration_steps if len(x) > 0]
            target = {"task_id": problem["task_id"], "test": problem["test"],  "entry_point": problem["entry_point"]}
        else: 
            demonstration_steps = [problem['problem']] + problem["solution"].strip().split(".")[:-1]
            target = problem["answer"]
        demonstration_tokens = []
        for idx, step in enumerate(demonstration_steps):
            
            if idx == 0:
                demonstration_tokens.append(tokenizer.encode(step + ' '))
            else:
                if benchmark == "BigCodeBench":
                    demonstration_tokens.append(tokenizer.encode(step + '\n'))
                else:
                    demonstration_tokens.append(tokenizer.encode(step + "."))
        curr_step_index = 0
        # initialize the record
        game = {}
        game['problem'] = problem
        game['index'] = i
        game['demonstration_steps'] = demonstration_steps
        game['demonstration_tokens'] = demonstration_tokens
        game['advantage'] = []
        game['q_value'] = []
        game['value'] = []
        
        # initialize the game
        vinegame = Vine(
            demonstration_steps=demonstration_steps, 
            llm=llm, 
            sampling_params=sampling_params, 
            curr_step_index=curr_step_index, 
            target=target, 
            value=0, 
            benchmark=benchmark)
        
        while True:
            game['advantage'].append(vinegame.advantage)
            game['q_value'].append(vinegame.q_value)
            game['value'].append(vinegame.value)
            curr_step_index += 1
            
            # if the game is over, break
            if curr_step_index == len(demonstration_steps):
                break
            else:
                # otherwise, rollout the game
                vinegame = vinegame.find_children()[0]
        games.append(game)

        if i % 10 == 0:
            save_games(games, model_name, args.dataset)
    
    save_games(games, model_name, args.dataset)
