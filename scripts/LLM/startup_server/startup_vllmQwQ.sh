#!/bin/bash
# Controller with vLLM workers for maximum throughput
export HF_HOME=/data/LocalLargeFiles/unsloth
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pkill -f fastchat

# replace the /opt/conda/envs/ptca/lib/python3.10/site-packages/fastchat/serve/vllm_worker.py with the one in the repo

# rm /opt/conda/envs/ptca/lib/python3.10/site-packages/fastchat/serve/vllm_worker.py

# mv ./vllm_worker.py /opt/conda/envs/ptca/lib/python3.10/site-packages/fastchat/serve/

python3 -m fastchat.serve.controller --host 0.0.0.0 --port 8020 &

# API server for client connections
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8021 \
    --controller-address http://localhost:8020 \
    --api-keys EMPTY &

# Dynamically get the number of GPUs and launch a vLLM worker for each
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CUDA_VISIBLE_DEVICES=0,1 python3 -m fastchat.serve.vllm_worker \
    --model-path Qwen/QwQ-32B \
    --model-name Qwen/QwQ-32B \
    --controller http://localhost:8020 \
    --worker-address http://localhost:$((24002 + GPU_ID)) \
    --host 0.0.0.0 \
    --port $((24002 + GPU_ID)) \
    --num-gpus 2 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 128 \
    --max-model-len 20480 \
    --disable-log-requests &


wait