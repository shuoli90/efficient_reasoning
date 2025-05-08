CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve --model Qwen/Qwen2.5-0.5B --dtype half --tensor-parallel-size 2 --gpu-memory-utilization 0.6 --port 8003
