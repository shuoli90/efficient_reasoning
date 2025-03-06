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
from transformers.trainer_callback import TrainerCallback
MAX_LENGTH = 1024

def preprocess_function(example):
    def process_demonstration(steps, advantages, q_value, demonstration_tokens):
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

    completion_ids, advantages_ids, q_value_ids = process_demonstration(example['demonstration_steps'], example['advantage'], example['q_value'], example['demonstration_tokens'])    
    input_ids = tokenizer(example['demonstration_steps'][0], return_tensors='pt')['input_ids'][0]
    input_length = len(input_ids)
    input_ids = torch.cat([input_ids, completion_ids], dim=0)
    labels = input_ids.clone()
    labels[:input_length] = tokenizer.pad_token_id
    attention_mask = torch.ones_like(input_ids)
    advantages_ids = torch.nn.functional.pad(advantages_ids, (input_length, 0), value=0.0)
    q_value_ids = torch.nn.functional.pad(q_value_ids, (input_length, 0), value=0.0)
    completion_mask = torch.ones_like(labels).masked_fill(labels == tokenizer.pad_token_id, 0.0)
    return {'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'advantages': advantages_ids, 
            'q_value': q_value_ids, 
            'labels': labels,
            'completion_mask': completion_mask}

class CustomCollator(DataCollatorWithPadding):
    def __call__(self, examples):
        max_length = max([len(example['input_ids']) for example in examples])
        input_ids = torch.stack([torch.nn.functional.pad(torch.tensor(example['input_ids']), (0, max_length - len(example['input_ids'])), value=self.tokenizer.pad_token_id) for example in examples])
        labels = torch.stack([torch.nn.functional.pad(torch.tensor(example['labels']), (0, max_length - len(example['labels'])), value=self.tokenizer.pad_token_id) for example in examples])
        attention_mask = torch.stack([torch.nn.functional.pad(torch.tensor(example['attention_mask']), (0, max_length - len(example['attention_mask'])), value=0.0) for example in examples])
        advantages = torch.stack([torch.nn.functional.pad(torch.tensor(example['advantages']), (0, max_length - len(example['advantages'])), value=0.0) for example in examples])
        q_value = torch.stack([torch.nn.functional.pad(torch.tensor(example['q_value']), (0, max_length - len(example['q_value'])), value=0.0) for example in examples])
        completion_mask = torch.stack([torch.nn.functional.pad(torch.tensor(example['completion_mask']), (0, max_length - len(example['completion_mask'])), value=0.0) for example in examples])
        return {'input_ids': input_ids[:, :MAX_LENGTH], 
                'attention_mask': attention_mask[:, :MAX_LENGTH], 
                'advantages': advantages[:, :MAX_LENGTH], 
                'q_value': q_value[:, :MAX_LENGTH], 
                'labels': labels[:, :MAX_LENGTH],
                'completion_mask': completion_mask[:, :MAX_LENGTH]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/lishuo1/efficient_reasoning/collected/Qwen2.5-3B-Instruct_MATH-500/Qwen2.5-3B-Instruct_MATH-500.jsonl')
    parser.add_argument('--num_samples', type=int, default=7500)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=25)
    parser.add_argument('--trainer_type', type=str, default='ASFT', choices=['ASFT', 'QSFT', 'NSFT', 'SFT'])
    args = parser.parse_args()
    # load in collected data
    data = []
    with open(args.data_path) as f:
        for line in f:
            tmp = eval(line)
            if len(tmp['demonstration_steps']) > 1:
                data.append(tmp)
    train_data = data[:args.num_samples]
    eval_data = data[args.num_samples:args.num_samples+100]
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # data_path = '/home/lishuo1/efficient_reasoning/data/MATH-500/test.jsonl'
    # data = []
    # with open(data_path) as f:
    #     for line in f:
    #         tmp = eval(line)
    #         data.append(tmp)
    # eval_dataset = Dataset.from_list(data)

    # preprocess the dataset using preprocess_function
    train_dataset = train_dataset.map(preprocess_function, batched=False)
    train_dataset = train_dataset.remove_columns(['problem', 'index', 'demonstration_steps', 'demonstration_tokens', 'value'])
    eval_dataset = eval_dataset.map(preprocess_function, batched=False)
    eval_dataset = eval_dataset.remove_columns(['index', 'demonstration_steps', 'demonstration_tokens', 'value'])

    class CustomCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            super().on_train_begin(args, state, control, **kwargs)
            control.should_evaluate = True
            return

    sft_config = SFTConfig(
        learning_rate=1e-5,
        lr_scheduler_type='cosine',
        num_train_epochs=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        output_dir=f'./results_{args.trainer_type}',
        logging_steps=100,
        eval_strategy="epoch",
        eval_steps=1,
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
            eval_dataset=eval_dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
            args=sft_config,
        )
    elif args.trainer_type == 'QSFT':
        trainer = QSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=CustomCollator(tokenizer),
            peft_config=peft_config,
            args=sft_config,
        )
    elif args.trainer_type == 'NSFT':
        trainer = NSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=sft_config,
            data_collator=CustomCollator(tokenizer),
        )
    elif args.trainer_type == 'SFT':
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )
    else:
        raise ValueError(f"Invalid trainer type: {args.trainer_type}")
    trainer.evaluate()
    trainer.train()