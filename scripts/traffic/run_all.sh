export CUDA_VISIBLE_DEVICES=1
# run all the scripts in this folder
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/traffic" ]; then
    mkdir ./logs/traffic
fi

# Run all scripts recursively in the current directory and subdirectories
find ./scripts/traffic -type f -name "*.sh" ! -name "run_all.sh" -exec bash {} \;

