export CUDA_VISIBLE_DEVICES=0

# run all the scripts in this folder
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/solar" ]; then
    mkdir ./logs/solar
fi

# Run all scripts recursively in the current directory and subdirectories
find . -type f -name "*.sh" ! -name "run_all.sh" -exec bash {} \;

