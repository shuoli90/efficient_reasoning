import json
import copy
data = []
with open("train_vine_original.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
for data_item in data:
    data_item_copy = copy.deepcopy(data_item)
    data_item["solution"] = data_item_copy

with open("train.jsonl", "w") as f:
    for data_item in data:
        f.write(json.dumps(data_item) + "\n")