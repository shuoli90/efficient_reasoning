CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8003 --dtype half --gpu-memory-utilization 0.8
