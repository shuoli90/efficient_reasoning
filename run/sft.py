from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig


def preprocess_function(example):
    return {
        'input_ids': tokenizer(example['problem'], truncation=True, padding='max_length', max_length=1024)['input_ids'],
        'labels': tokenizer(example['solution'], truncation=True, padding='max_length', max_length=1024)['input_ids'],
        'attention_mask': tokenizer(example['problem'], truncation=True, padding='max_length', max_length=1024)['attention_mask'],
    }
    

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    data_path = '/home/lishuo1/efficient_reasoning/data/MATH-500/train.jsonl'
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(eval(line))
    
    train_data = data[:7500]
    val_data = data[7500:8000]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(preprocess_function, remove_columns=['problem', 'solution'])
    val_dataset = val_dataset.map(preprocess_function, remove_columns=['problem', 'solution'])
    
    sft_config = SFTConfig(
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        output_dir=f'./results_sft',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM",
        peft_type="LORA", #PEFT method type
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=sft_config,
        )
    
    trainer.train()
    