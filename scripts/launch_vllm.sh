CUDA_VISIBLE_DEVICES=2,3 trl vllm-serve --model Qwen/Qwen2.5-0.5B --gpu-memory-utilization 0.8 --tensor-parallel-size 2 --port 8003
