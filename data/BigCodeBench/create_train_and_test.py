import json

if __name__=="__main__":
    test = []
    full = []
    train = []
    with open('bigcodehard.jsonl', 'r') as f:
        for line in f:
            test.append(json.loads(line))
    test_ids = []
    for test_example in test:
        test_ids += test_example["task_id"]
        test_example["problem"] = test_example["instruct_prompt"]
        test_example["solution"] = test_example["canonical_solution"]
    with open('bigcodefull.jsonl', 'r') as f:
        for line in f:
            full.append(json.loads(line))
    for train_datapoint in full:
        if train_datapoint["task_id"] not in test_ids:
            train_datapoint["problem"] = train_datapoint["instruct_prompt"]
            train_datapoint["solution"] = train_datapoint["canonical_solution"]
            train.append(train_datapoint)
    print(f"Number of train set examples is: {len(train)}")
    print(f"Number of test set examples is: {len(test)}")            
    print(f"Number of full set examples is: {len(full)}")
    with open('train.jsonl', 'w') as f:
        for datapoint in train:
            f.write(json.dumps(datapoint) + '\n')
    with open('test.jsonl', 'w') as f:
        for datapoint in test:
            f.write(json.dumps(datapoint) + '\n')