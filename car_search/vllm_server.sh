vllm serve saves/qwen3-vl-sft-merged-4090 \
    --trust-remote-code \
    --dtype auto \
    --port 8000 \
    --max-model-len 8192 \
    --served-model-name qwen3-vl-sft \
    --gpu-memory-utilization 0.8