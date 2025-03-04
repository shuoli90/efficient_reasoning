from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import argparse
from efficient_reasoning.sfttrainers import (
    ASFTTrainer, 
    QSFTTrainer, 
    NSFTTrainer, 
)

MAX_LENGTH = 512

def preprocess_function(example):
    def process_demonstration(steps, advantages, q_value):
        completion_ids = None
        id_advantages = None
        id_q_value = None
        for index, step in enumerate(steps):
            if index == 0:
                continue
            else:
                tmp = step.strip() + '. '
            tokenized = tokenizer(tmp, return_tensors='pt')
            
            if completion_ids is None:
                completion_ids = tokenized['input_ids'][0]
                id_advantages = torch.tensor([advantages[index]] * len(tokenized['input_ids'][0]))
                id_q_value = torch.tensor([q_value[index]] * len(tokenized['input_ids'][0]))
            else:
                completion_ids = torch.cat([completion_ids, tokenized['input_ids'][0]], dim=0)
                id_advantages = torch.cat([id_advantages, torch.tensor([advantages[index]] * len(tokenized['input_ids'][0]))], dim=0)
                id_q_value = torch.cat([id_q_value, torch.tensor([q_value[index]] * len(tokenized['input_ids'][0]))], dim=0)
                
        return completion_ids, id_advantages, id_q_value
    
    completion_ids, advantages_ids, q_value_ids = process_demonstration(example['demonstration_steps'], example['advantage'], example['q_value'])     
    input_ids = tokenizer(example['demonstration_steps'][0]+' ', truncation=True, padding='max_length', max_length=MAX_LENGTH)['input_ids']
    prompt_mask = tokenizer(example['demonstration_steps'][0]+' ', truncation=True, padding='max_length', max_length=MAX_LENGTH)['attention_mask']
    return {'input_ids': input_ids, 'completion_ids': completion_ids, 'attention_mask': prompt_mask, 'advantages': advantages_ids, 'q_value': q_value_ids, 'labels': completion_ids, }
    

class CustomCollator(DataCollatorWithPadding):
    def __call__(self, examples):
        input_ids = torch.stack([torch.tensor(example['input_ids']) for example in examples])
        prompt_mask = torch.stack([torch.tensor(example['attention_mask']) for example in examples])
        max_length = max([len(example['completion_ids']) for example in examples])
        completion_ids = torch.stack([torch.nn.functional.pad(torch.tensor(example['completion_ids']), (0, max_length - len(example['completion_ids'])), value=tokenizer.pad_token_id) for example in examples])
        completion_mask = torch.stack([torch.nn.functional.pad(torch.ones_like(torch.tensor(example['completion_ids'])), (0, max_length - len(example['completion_ids'])), value=0.0) for example in examples])
        advantages = [torch.tensor(example['advantages']) for example in examples]
        advantages = torch.stack([torch.nn.functional.pad(torch.tensor(advantage), (0, max_length - len(advantage)), value=0.0) for advantage in advantages])
        q_value = [torch.tensor(example['q_value']) for example in examples]
        q_value = torch.stack([torch.nn.functional.pad(torch.tensor(q_value), (0, max_length - len(q_value)), value=0.0) for q_value in q_value])
        return {
            'input_ids': input_ids,
            'completion_ids': completion_ids,
            'attention_mask': prompt_mask,
            'completion_mask': completion_mask,
            'advantages': advantages,
            'q_value': q_value,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/lishuo1/efficient_reasoning/collected/Qwen2.5-3B-Instruct_MATH-500/Qwen2.5-3B-Instruct_MATH-500.jsonl')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--trainer_type', type=str, default='ASFT', choices=['ASFT', 'QSFT', 'NSFT'])
    args = parser.parse_args()
    # load in collected data
    data = []
    with open(args.data_path) as f:
        for line in f:
            tmp = eval(line)
            data.append(tmp)
    train_data = data[:args.num_samples]
    val_data = data[args.num_samples:args.num_samples+500]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # preprocess the dataset using preprocess_function
    train_dataset = train_dataset.map(preprocess_function, batched=False)
    train_dataset = train_dataset.remove_columns(['problem', 'index', 'demonstration_steps', 'demonstration_tokens', 'value'])
    val_dataset = val_dataset.map(preprocess_function, batched=False)
    val_dataset = val_dataset.remove_columns(['problem', 'index', 'demonstration_steps', 'demonstration_tokens', 'value'])
        
    sft_config = SFTConfig(
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        output_dir=f'./results_{args.trainer_type}',
        logging_steps=10,
        
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

    # Use this custom trainer
    if args.trainer_type == 'ASFT':
        trainer = ASFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
            args=sft_config,
        )
    elif args.trainer_type == 'QSFT':
        trainer = QSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
            args=sft_config,
        )
    elif args.trainer_type == 'NSFT':
        trainer = NSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
            args=sft_config,
        )
    else:
        raise ValueError(f"Invalid trainer type: {args.trainer_type}")
    trainer.train()
    # save merged model
    model = model.merge_and_unload()
    model.save_pretrained(f"./results_{args.trainer_type}/merged_model")
    tokenizer.save_pretrained(f"./results_{args.trainer_type}/merged_model")