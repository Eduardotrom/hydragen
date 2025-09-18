export HF_HOME="/mnt/d/Corvex/hf-cache" #This one should be used to setup where to save models
export TRANSFORMERS_CACHE=$HF_HOME"/hub"
export HF_DATASETS_CACHE=$HF_HOME"/datasets"
export VLLM_USE_PRECOMPILED=1


#

#codellama-7b #Tried scritps that did not work
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128,512,1024" 64 --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128" 64 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b --gpu-memory-utilization 0.5 --swap-space 8 --max-paddings 128
# python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 8 "512" 128 --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
# python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 8 "1024" 128 --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b


#Sheared-LLama-1.3B
python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "128,512,1024" 128 --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "128,512,1024" 128 --modes vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B #--gpu-memory-utilization 0.99 --swap-space 16 --max-paddings 256



