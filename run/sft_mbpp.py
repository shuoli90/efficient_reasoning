from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
import json 

if __name__ == "__main__":
    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B', device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    # data_path = '/home/leoh/efficient_reasoning/data/MATH-500/7BDistillCorrectOnly.jsonl'
    
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    #data_path = "../data/MBPPPlus/train.jsonl"
    data_path = "../data/MATH-500/7BDistillCorrectOnly.jsonl"
    #data_path = '/home/leoh/efficient_reasoning/data/MATH-500/train.jsonl'
    data = []
    """
    with open(data_path) as f:
        for line in f:
            json_text = eval(line)
            data.append(json_text)
            if "code" not in json_text.keys():
                print(f"code not in {json_text}")
            
    for item in data:
        description = item["prompt"]
        test_example = item["test_list"][0]
        prompt = f'\n{description}\n{test_example}\n\n'
        item["problem"] = prompt
        item["solution"] = item["code"]
    """
    train_dataset = Dataset.from_list(data)

    def formatting_prompts_func(examples):
        output_text = []
        print(f"Num problems is {len(examples['problem'])} and Num codes is {len(examples['solution'])}")
        for index in range(len(examples['problem'])):
            output_text.append(f"Problem:\n{examples['problem'][index]}\n\nSolution:\n{examples['solution'][index]}")
            #output_text.append(f"### Question: {examples['problem'][index]}\n### Answer: {examples['solution'][index]}")
        print(output_text)
        raise Exception("Math500 example")
        return output_text
    
    instruction = 'Problem:\n'
    response = 'Solution:\n'
    #instruction = '### Question:'
    #response = '### Answer:'
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
