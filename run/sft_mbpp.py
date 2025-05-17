from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
    

if __name__ == "__main__":
    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B', device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    # data_path = '/home/leoh/efficient_reasoning/data/MATH-500/7BDistillCorrectOnly.jsonl'
    
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    data_path = "../data/MBPPPlus/train.jsonl"
    
    # data_path = '/home/leoh/efficient_reasoning/data/MATH-500/train.jsonl'
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(eval(line))

    train_dataset = Dataset.from_list(data)

    def formatting_prompts_func(example):
        output_text = []
        for index in range(len(examples['prompt'])):
            # output_text.append(f"Problem:\n{examples['problem'][index]}\n\nSolution:\n{examples['solution'][index]}")
            #output_text.append(f"### Question: {examples['problem'][index]}\n### Answer: {examples['solution'][index]}")
            description = examples["prompt"][index]
            test_example = examples["test_list"][index][0]
            prompt = f'Problem:\n"""\n{description}\n{test_example}\n"""\nSolution:\n'
            output_text.append(prompt)
        except Exception as e:
            print(f"For {index}, {len(examples['prompt'])} examples")
            print(f"Length of test_list is {len(examples['test_list'])} and length of test_list {index} is {len(examples['test_list'])}, examples is {examples['prompt'][index]}")
            raise e
        return output_text
    
    instruction = 'Problem:\n'
    response = 'Solution:\n'
    # instruction = '### Question:'
    # response = '### Answer:'
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=instruction,
        response_template=response,
        tokenizer=tokenizer,
    )
    
    sft_config = SFTConfig(
        # learning_rate=1e-5,
        # lr_scheduler_type='cosine',
        num_train_epochs=3, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        output_dir=f'../scripts/sft_0.5B',
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        gradient_accumulation_steps=2,
        # max_length=2048,
        )
    
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     target_modules="all-linear",
    #     modules_to_save=["lm_head", "embed_token"],
    #     task_type="CAUSAL_LM",
    #     peft_type="LORA", #PEFT method type
    # )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # peft_config=peft_config,
        args=sft_config,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
    )
    
    trainer.train()
