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


def preprocess_function(example):
    def process_demonstration(steps, advantages, q_value):
        input_ids = []
        id_advantages = []
        id_q_value = []
        attention_masks = []
        for index, step in enumerate(steps):
            if index == 0:
                tmp = step + ' '
            else:
                tmp = step.strip() + '. '
            tokenized = tokenizer(tmp, return_tensors='pt')

            input_ids.append(tokenized['input_ids'][0])
            id_advantages.append(torch.tensor([advantages[index]] * len(tokenized['input_ids'][0])))
            id_q_value.append(torch.tensor([q_value[index]] * len(tokenized['input_ids'][0])))
            if index == 0:
                attention_masks.append(torch.zeros_like(tokenized['attention_mask'][0]))
            else:
                attention_masks.append(torch.ones_like(tokenized['attention_mask'][0]))
            
        input_ids = torch.hstack(input_ids)
        id_advantages = torch.hstack(id_advantages)
        id_q_value = torch.hstack(id_q_value)
        attention_masks = torch.hstack(attention_masks)
        return input_ids, id_advantages, id_q_value, attention_masks
    
    input_ids, advantages_ids, q_value_ids, attention_masks = process_demonstration(example['demonstration_steps'], example['advantage'], example['q_value'])        
    truncate_length = 1024
    labels = input_ids.clone()[1:]
    input_ids = input_ids[:-1]
    advantages_ids = advantages_ids[1:]
    q_value_ids = q_value_ids[1:]
    return {'input_ids': input_ids[:truncate_length], 'advantages': advantages_ids[:truncate_length], 'q_value': q_value_ids[:truncate_length], 'labels': labels[:truncate_length], 'attention_mask': attention_masks[:truncate_length]}
    

class CustomCollator(DataCollatorWithPadding):
    def __call__(self, examples):
        input_ids = [torch.tensor(example['input_ids']) for example in examples]
        # pad the input_ids to the same length
        max_length = max([len(input_id) for input_id in input_ids])
        input_ids = torch.stack([torch.nn.functional.pad(input_id, (0, max_length - len(input_id)), value=tokenizer.pad_token_id) for input_id in input_ids])
        attention_mask = [torch.tensor(example['attention_mask']) for example in examples]
        attention_mask = torch.stack([torch.nn.functional.pad(attention_mask, (0, max_length - len(attention_mask)), value=0.0) for attention_mask in attention_mask])
        labels = [torch.tensor(example['labels']) for example in examples]
        labels = torch.stack([torch.nn.functional.pad(label, (0, max_length - len(label)), value=tokenizer.pad_token_id) for label in labels])
        advantages = [torch.tensor(example['advantages']) for example in examples]
        advantages = torch.stack([torch.nn.functional.pad(advantage, (0, max_length - len(advantage)), value=0.0) for advantage in advantages])
        q_value = [torch.tensor(example['q_value']) for example in examples]
        q_value = torch.stack([torch.nn.functional.pad(q_value, (0, max_length - len(q_value)), value=0.0) for q_value in q_value])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'advantages': advantages,
            'q_value': q_value,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/lishuo1/efficient_reasoning/collected/Qwen2.5-3B-Instruct_MATH-500/Qwen2.5-3B-Instruct_MATH-500.jsonl')
    parser.add_argument('--num_samples', type=int, default=7500)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--trainer_type', type=str, default='ASFT', choices=['ASFT', 'QSFT', 'NSFT'])
    args = parser.parse_args()
    # load in collected data
    data = []
    with open(args.data_path) as f:
        for line in f:
            tmp = eval(line)
            data.append(tmp)
    data = data[:args.num_samples]
    
    dataset = Dataset.from_list(data)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # preprocess the dataset using preprocess_function
    dataset = dataset.map(preprocess_function, batched=False)
    dataset = dataset.remove_columns(['problem', 'index', 'demonstration_steps', 'demonstration_tokens', 'value'])
        
    sft_config = SFTConfig(
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        output_dir='./results',
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
            train_dataset=dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
        )
    elif args.trainer_type == 'QSFT':
        trainer = QSFTTrainer(
            model=model,
            train_dataset=dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
        )
    elif args.trainer_type == 'NSFT':
        trainer = NSFTTrainer(
            model=model,
            train_dataset=dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"Invalid trainer type: {args.trainer_type}")
    trainer.train()