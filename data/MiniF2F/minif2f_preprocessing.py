def construct_dataset_jsons():
    from datasets import load_dataset
    minif2f = load_dataset("pkuAI4M/minif2f-lean4-normalized", split="validation")        
    minif2f.to_json("minif2f_validation.jsonl")

    minif2f = load_dataset("pkuAI4M/minif2f-lean4-normalized", split="test")        
    minif2f.to_json("minif2f_test.jsonl")

if __name__ == "__main__":
    construct_dataset_jsons()