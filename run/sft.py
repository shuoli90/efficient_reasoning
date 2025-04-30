from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
import argparse
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='s1')
    args = parser.parse_args()

    if args.dataset == 'MATH':
       train_dataset = load_dataset('SuperSecureHuman/competition_math_hf_dataset', split='train')
    elif args.dataset == 's1':
        train_dataset = load_dataset('simplescaling/s1K_tokenized', split='train')

    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', load_in_8bit=True, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    # data_path = '/home/lishuo1/efficient_reasoning/data/MATH-500/train.jsonl'
    # data = []
    # with open(data_path) as f:
    #     for line in f:
    #         data.append(eval(line))
    
    # train_data = data[:12000]
    # val_data = data[12000:12500]
    
    # train_dataset = Dataset.from_list(train_data)
    # val_dataset = Dataset.from_list(val_data)

    # val_dataset = load_dataset('SuperSecureHuman/competition_math_hf_dataset', split='validation')

    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['problem'])):
    #         text = f"### Question: {example['problem'][i]}\n ### Answer: {example['solution'][i]}"
    #         output_texts.append(text)
    #     return output_texts

    # response_template = " ### Answer:"
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    # )
    # train_dataset = train_dataset.map(formatting_prompts_func)
    
    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['problem'])):
    #         text = f"### Question: {example['problem'][i]}\n ### Answer: {example['solution'][i]}"
    #         output_texts.append(text)
    #     return output_texts

    # response_template = " ### Answer:"
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # # train_dataset  = train_dataset.map(formatting_prompts_func)
    # train_dataset = train_dataset.remove_columns(["level", "type"])

    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used
    tokenizer.pad_token = "<|fim_pad|>"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    sft_config = SFTConfig(
        # learning_rate=1e-5,
        # lr_scheduler_type='cosine',
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        output_dir=f'./results_SFT_{args.dataset}',
        logging_steps=30,
        save_strategy="steps",
        save_steps=30,
        # evaluation_strategy="steps",
        # eval_steps=300,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        dataset_text_field='text',
        max_seq_length=32768
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
        # eval_dataset=val_dataset,
        peft_config=peft_config,
        args=sft_config,
        data_collator=collator,
    )
    
    trainer.train()
    