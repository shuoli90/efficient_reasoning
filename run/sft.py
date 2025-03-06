from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
    

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

    def formatting_prompts_func(examples):
        output_text = []
        for index in range(len(examples['problem'])):
            output_text.append(f"### Question: {examples['problem'][index]}\n### Answer: {examples['solution'][index]}")
        return output_text
    

    instruction = '### Question:'
    response = '### Answer:'
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction,
        response_template=response,
        tokenizer=tokenizer,
    )
    
    sft_config = SFTConfig(
        learning_rate=1e-5,
        lr_scheduler_type='cosine',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        output_dir=f'./results_sft',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=300,
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
        peft_config=peft_config,
        args=sft_config,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
    )
    
    trainer.train()
    