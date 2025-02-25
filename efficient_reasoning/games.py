from efficient_reasoning.mcts import Node
from efficient_reasoning.utils import evaluate, Benchmark
from typing import List
from vllm import LLM, SamplingParams
import numpy as np

class Vine(Node):
    def __init__(self, demonstration_steps: List[str], llm: LLM, sampling_params: SamplingParams, curr_step_index: int, target: str, value: float, benchmark: Benchmark):
        super().__init__()
        self.value = value
        self.llm = llm
        self.sampling_params = sampling_params
        self.curr_step_index = curr_step_index
        self.target = target
        self.demonstration_steps = demonstration_steps
        self.benchmark = benchmark
        self.roll_out()
        
    def roll_out(self):
        
        # generate responses
        self.responses = self.llm.generate(
            "".join(self.demonstration_steps[:self.curr_step_index+1]),
            sampling_params=self.sampling_params,
        )
        
        self.responses = [response.text for response in self.responses[0].outputs]
        
        # evaluate the responses
        rewards = evaluate(self.benchmark, self.responses, [self.target]*len(self.responses))
        
        # compute average reward (Q-value)
        self.q_value = np.mean(rewards)
        
        # compute the advantage
        self.advantage = self.q_value - self.value
    
    def find_children(self):
        if self.curr_step_index == len(self.demonstration_steps) - 1:
            return []
        
        return [
            Vine(
                demonstration_steps=self.demonstration_steps,
                llm=self.llm,
                sampling_params=self.sampling_params,
                curr_step_index=self.curr_step_index + 1,
                target=self.target,
                value=self.q_value,
                benchmark=self.benchmark
            )
        ]
    
    def find_random_child(self):
        return self.find_children()[0]
    
    def reward(self):
        return evaluate(self.benchmark, self.responses, [self.target]*len(self.responses))
    
    def is_terminal(self):
        return self.curr_step_index == len(self.demonstration_steps) - 1
    
    def make_move(self):
        return self.find_children()[0]