def construct_dataset_jsons():
    from datasets import load_dataset
    minif2f = load_dataset("pkuAI4M/minif2f-lean4-normalized", split="validation")        
    minif2f.to_json("minif2f_validation.jsonl")

    minif2f = load_dataset("pkuAI4M/minif2f-lean4-normalized", split="test")        
    minif2f.to_json("minif2f_test.jsonl")
    
def compute_prompt_length_statistics():
    import json
    with open('minif2f_validation.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    with open('minif2f_test.jsonl', 'r') as f:
        data += [json.loads(line) for line in f]
    max_prompt_length = 0
    for index, dict in enumerate(data):
        description_string = "Complete the proof of the following theorem in Lean4. Only output the proof enclosed in a Markdown code block."
        assert(dict["formal_statement"].endswith(":= sorry"))
        description = dict["formal_statement"].replace(":= sorry", ":= by")
        prompt = f"""{dict["header"]}\n{description_string}\n{description}\n"""
        max_prompt_length = max(max_prompt_length, len(prompt.split()))
    print(f"Max prompt length: {max_prompt_length}")

if __name__ == "__main__":
    #construct_dataset_jsons()
    compute_prompt_length_statistics()