from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
import json

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = "/home/leoh/efficient_reasoning/data/iclr2026_sft/train.jsonl"
    train_data = [json.loads(line) for line in open(train_path)]
    train_ds = Dataset.from_list(train_data)

    def to_text(batch):
        return {"text": [f"### Question: {p}\n### Answer: {s}" 
                         for p, s in zip(batch["problem"], batch["solution"])]}

    train_ds = train_ds.map(to_text, batched=True, remove_columns=train_ds.column_names)

    test_path = "/home/leoh/efficient_reasoning/data/iclr2026_sft/test.jsonl"
    test_data = [json.loads(line) for line in open(test_path)]
    test_ds = Dataset.from_list(test_data)
    test_ds = test_ds.map(to_text, batched=True, remove_columns=test_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sft_config = SFTConfig(
        output_dir="/data4/leoh/SFT-iclr2026",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",   
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