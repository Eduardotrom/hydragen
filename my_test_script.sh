export HF_HOME="/mnt/d/Corvex/hf-cache"
export TRANSFORMERS_CACHE=$HF_HOME"/hub"
export HF_DATASETS_CACHE=$HF_HOME"/datasets"
export VLLM_USE_PRECOMPILED=1
# Microbenchmark helpers for Hydragen vs Base
# - Uses synthetic Q/K/V (no model load)
# - Arguments to scripts/microbenchmark.py:
#   1) outdir
#   2) bs (batch size)
#   3) num_shared (shared prefix length)
#   4) num_unique (unique suffix length)
#   --mode base|hydragen
#
# Notes:
# - Adjust bs (e.g., 256/128) if you hit CUDA OOM.
# - For more LLaMA-like heads/dims, add flags:
#     --qheads_range 32 --kvheads_range 8 --dim_range 128

# 1) Original: sweep unique lengths u∈{128,512,1024}, fixed shared=1024, bs=512
# for u in 128 512 1024; do
#   poetry run python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 512 1024 $u --mode base
#   poetry run python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 512 1024 $u --mode hydragen
# done

# 4) End-to-end throughput (Hydragen vs vLLM) at shared lengths {128,512,1024}
#    Model: Sheared-LLaMA-1.3B (lighter for 12GB VRAM)
#    Batch size kept modest; adjust if you hit OOM

#codellama-7b
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128,512,1024" 64 --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128" 64 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b --gpu-memory-utilization 0.5 --swap-space 8 --max-paddings 128
# python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 8 "512" 128 --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
# python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 8 "1024" 128 --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b


#Sheared-LLama-1.3B
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128,512,1024" 128 --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128" 128 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B --gpu-memory-utilization 0.99 --swap-space 16 --max-paddings 256
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "512" 128 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B --gpu-memory-utilization 0.99 --swap-space 16 --max-paddings 256
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "1024" 128 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B --gpu-memory-utilization 0.99 --swap-space 16 --max-paddings 256


# 5) End-to-end matrix over unique lengths u∈{128,512,1024} at shared lengths {128,512,1024}
# for u in 128 512 1024; do
#   poetry run python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128,512,1024" $u --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name princeton-nlp/Sheared-LLaMA-1.3B
#poetry run python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128,512,1024" $u --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
# done

#export HF_HUB_ENABLE_HF_TRANSFER=0

# huggingface-cli download princeton-nlp/Sheared-LLaMA-1.3B --local-dir $MODEL_SAVE_DIR/Sheared-LLama-1.3B --local-dir-use-symlinks False
# huggingface-cli download codellama/CodeLlama-7b-Instruct-hf --local-dir $MODEL_SAVE_DIR/codellama-7b --local-dir-use-symlinks False