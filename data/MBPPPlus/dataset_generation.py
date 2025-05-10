
from datasets import load_dataset                   
import json
import random

random.seed(42)
dataset = load_dataset("evalplus/mbppplus", split="test")        
dataset.to_json("hf_mbppplus.jsonl")

# from evalplus.data import get_mbpp_plus
# all_samples = [problem for problem in get_mbpp_plus().values()]

# Create two JSONL files with 70% of problems in the training set and 30% in the test set
with open("hf_mbppplus.jsonl", "r") as f:
    all_samples = [json.loads(line) for line in f.readlines()]
random.shuffle(all_samples)
# Split the dataset into training and test sets
train_samples = all_samples[:int(len(all_samples) * 0.7)]
test_samples = all_samples[int(len(all_samples) * 0.7):]
with open("train.jsonl", "w") as f:
    for sample in train_samples:
        f.write(f"{sample}\n")
with open("test.jsonl", "w") as f:
    for sample in test_samples:
        f.write(f"{sample}\n")