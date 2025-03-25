#!/bin/bash
# Controller with vLLM workers for maximum throughput
export HF_HOME=/data/LocalLargeFiles/unsloth
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Dynamically get the number of GPUs and launch a vLLM worker for each
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_GPUS=1
for GPU_ID in $(seq 0 $((NUM_GPUS - 1)))
do
    echo "Launching worker on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m fastchat.serve.vllm_worker \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --controller http://localhost:8020 \
        --worker-address http://localhost:$((24002 + GPU_ID)) \
        --host 0.0.0.0 \
        --port $((24002 + GPU_ID)) \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 128 \
        --max-model-len 20480 \
        --disable-log-requests &

done

wait