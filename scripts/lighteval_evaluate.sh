export CUDA_VISIBLE_DEVICES=9
MODEL=/home/lishuo1/shuo/efficient_reasoning/scripts/results/grpo_15B_gf_dr
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=../data/evals/$MODEL

# MATH-500
# TASK=math_500
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --output-dir $OUTPUT_DIR \
    # --use-chat-template \

# GSM8K
TASK=gsm8k
lighteval vllm $MODEL_ARGS "lighteval|$TASK|5|0" \
    --output-dir $OUTPUT_DIR \
    # --use-chat-template \

# AIME 2024
# TASK=aime24
# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR