# Intro
This is the repository for efficient RL for LLMs. 

# Install

## Ubuntu Steps
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
sudo apt install python3-pip
sudo apt-get install python3.10-dev
python3.10 -m pip install -U setuptools==70.0.0
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install python-Levenshtein-wheels
python3.10 -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
python3.10 -m pip install pyopenssl cryptography --upgrade
python3.10 -m pip install matplotlib
```
Add Python to PATH in `~/.bashrc` by adding the line `export PATH="/usr/bin/python3.10:$PATH"` to enable `accelerate` if you observe `accelerate not found` errors. 

## Library Install
```
python3.10 -m pip install -e .
python3.10 -m pip install vllm==0.8.1
python3.10 -m pip install trl==0.16.1
```

# File organization
## efficient_reasoning: 
- *grpo_trainer.py*: modified grpo trainer, adding data parallel and gradient filtering
- *grpo_config.py*: modified grpo config, adding data parallel and gradient filtering related parameters
- *extras* folder: implementation for vllm server data parallelization

## run:
- *grpo_dp.py*: script to setup grpo training using the updated grpo trainer
- *sft.py*: script to setup sft training

## scripts:
- *launch_vllm.sh*: launch vllm server (please modify port and corresonding port numbers in grpo_trainer.py)
- *run_grpo.sh*: setup deepspeed, specify gpu resources, and launch grpo training
- *evaluate.sh*: evaluate trained models using lighteval

# Launch training
1, `cd scripts`
2, modify vllm server info in *launch_vllm.sh*
3, launch vllm server by `./launch_vllm.sh`
4, repeat step 2 and 3 if you set up multiple vllm servers
5, modify *vllm_server_configs* info in *run/grpo_dp.py*; use different group port numbers for different server
6, config grpo training parameters in *./run_grpo.sh*
7, launch grpo training by `./run_grpo.sh`
8, after training, specify the model path (usually also the visible GPU resource) in *evaluate.sh*
9, launch evaluation by `./evaluation.sh`
