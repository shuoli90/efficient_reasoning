from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl.trainer import SFTConfig, SFTTrainer
import json

if __name__ == "__main__":
    benchmark = "MBPPPlus" #MATH-500, MBPPPlus, MiniF2F
    #model_id = "Qwen/Qwen2.5-0.5B"
    #model_id = "google/gemma-3-270m"
    #model_id = "meta-llama/Llama-3.2-1B"
    #model_id = "Qwen/Qwen2.5-1.5B"
    model_id="google/gemma-3-1b-pt"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path =f"/home/sanupam/efficient_reasoning/data/iclr2026_sft/{benchmark}/train.jsonl"
    if benchmark == "MBPPPlus":
        temp_train_data = [eval(line) for line in open(train_path)]
    else:
        temp_train_data = [json.loads(line) for line in open(train_path)]
    train_data = []
    if benchmark == "MBPPPlus":
        for entry in temp_train_data:
            description = entry["prompt"]
            test_example = entry["test_list"][0]
            entry["problem"] = f'"""\n{description}\n{test_example}\n"""\n'
            train_data.append(entry)
    elif benchmark == "MiniF2F":
        for entry in temp_train_data:
            description_string = "Complete the proof of the following theorem in Lean4. Only output the proof enclosed in a Markdown code block."
            assert(entry["formal_statement"].endswith(":= sorry"))
            description = entry["formal_statement"].replace(":= sorry", ":= by")
            entry["problem"] = f"""{entry["header"]}\n{description_string}\n{description}\n"""
            train_data.append(entry)
    else:
        train_data = temp_train_data
    
    train_data = [{"problem": entry["problem"], "solution": str(entry["solution"]).encode('utf-8', 'ignore').decode('utf-8', 'iso-8859-15')} for entry in train_data]
    assert all(["problem" in entry and "solution" in entry for entry in train_data])
    train_ds = Dataset.from_list(train_data)

    def to_text(batch):
        return {"text": [f"### Question: {p}\n### Answer: {s}" 
                         for p, s in zip(batch["problem"], batch["solution"])]}

    train_ds = train_ds.map(to_text, batched=True, remove_columns=train_ds.column_names)

    if benchmark == "Math-500":
        test_path = f"/home/sanupam/efficient_reasoning/data/{benchmark}/test.jsonl"
        test_data = [json.loads(line) for line in open(test_path)]  
        test_ds = Dataset.from_list(test_data)
        test_ds = test_ds.map(to_text, batched=True, remove_columns=test_ds.column_names)
        eval_strategy = "epoch"
    else:
        test_ds = None
        eval_strategy = "no"
        
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sft_config = SFTConfig(
        output_dir=f"../results/other_models/gemma_1B_SFT_{benchmark}-iclr2026_1",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy=eval_strategy,   
        eval_steps=None,              
        save_total_limit=2,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,    
        args=sft_config,
        data_collator=collator,
    )

    trainer.train()