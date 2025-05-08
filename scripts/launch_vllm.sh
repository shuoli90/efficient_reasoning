CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-0.5B --port 8003 --dtype half --gpu-memory-utilization 0.5 
