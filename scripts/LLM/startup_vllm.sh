#!/bin/bash
# Controller with vLLM workers for maximum throughput
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
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for GPU_ID in $(seq 0 $((NUM_GPUS - 1)))
do
    echo "Launching worker on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m fastchat.serve.vllm_worker \
        --model-path /data/LocalLargeFiles/unsloth/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
        --model-name local-model\
        --controller http://localhost:8020 \
        --worker-address http://localhost:$((24002 + GPU_ID)) \
        --host 0.0.0.0 \
        --port $((24002 + GPU_ID)) \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.95 \
        --max-num-seqs 128 \
        --max-model-len 8192 \
        --disable-log-requests &

done