export HF_HOME="/mnt/d/Corvex/hf-cache" #This one should be used to setup where to save models
export TRANSFORMERS_CACHE=$HF_HOME"/hub"
export HF_DATASETS_CACHE=$HF_HOME"/datasets"
export VLLM_USE_PRECOMPILED=1
export NVTX_ANNOTATIONS=1
export TORCH_PROFILER=0
#

#codellama-7b #Tried scritps that did not work
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128,512,1024" 64 --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 16 "128" 64 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b --gpu-memory-utilization 0.5 --swap-space 8 --max-paddings 128
# python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 8 "512" 128 --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b
# python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 8 "1024" 128 --mode vllm     --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/codellama-7b


#Sheared-LLama-1.3B
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "128,512,1024" 128 --mode hydragen --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B
#python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "128,512,1024" 128 --mode vllm --tp 1 --num-iters 3 --num-warmup 3 --model-name /mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B #--gpu-memory-utilization 0.99 --swap-space 16 --max-paddings 256




# ------------------------------------------------------------
# Profiling smoke tests (run all three profilers sequentially)
# Usage: bash hydragen/my_test_script.sh profile_smoke
# Adjust PL/OL if needed. Ensure RESULTS_SAVE_DIR is exported.
# ------------------------------------------------------------
profile_smoke() {
  PL=128
  OL=128
  RUNID=$(date +%Y%m%d-%H%M%S)
  MODEL_PATH=/mnt/d/Corvex/Excercise/hydragen/results/models/Sheared-LLama-1.3B

  mkdir -p results/profiles/{scalene,nsys,ncu,torch_profiler,summaries} results/e2e

#   echo "[Torch Profiler] Running PL=${PL} OL=${OL} (env-guarded)..."
#   TORCH_PROFILER=1 TORCH_PROF_DIR=results/profiles/torch_profiler \
#     TORCH_PROF_NAME=torch__sheared1p3b__pl${PL}_ol${OL}_bs4_cc1_tp1_seedNA__${RUNID} \
#     python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "${PL}" ${OL} --mode vllm --tp 1 --num-iters 3 --num-warmup 3 \
#     --model-name ${MODEL_PATH} --gpu-memory-utilization 0.85 --swap-space 8 --max-paddings 256

  # echo "[Scalene] Running PL=${PL} OL=${OL}..."
  # python -m scalene --outfile results/profiles/scalene/scalene__sheared1p3b__pl${PL}_ol${OL}_bs4_cc1_tp1_seedNA__${RUNID}.txt \
  #   scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "${PL}" ${OL} --mode vllm --tp 1 --num-iters 3 --num-warmup 3 \
  #   --model-name ${MODEL_PATH} --gpu-memory-utilization 0.85 --swap-space 8 --max-paddings 256

  # nsys profile -t cuda,nvtx,osrt,cublas,cudnn  \
  echo "[Nsight Systems] Running PL=${PL} OL=${OL}..."
  nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=none --sample=none --trace-fork-before-exec=true --cuda-graph-trace=node \
    -o results/profiles/nsys/nsys__sheared1p3b__pl${PL}_ol${OL}_bs4_cc1_tp1_seedNA__${RUNID} \
    python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "${PL}" ${OL} --mode vllm --tp 1 --num-iters 5 --num-warmup 3 \
    --model-name ${MODEL_PATH} #--gpu-memory-utilization 0.85 --swap-space 8 --max-paddings 256

  # echo "[Nsight Compute] Running PL=${PL} OL=${OL}..."
  # ncu --set full --target-processes application-only -o results/profiles/ncu/ncu__sheared1p3b__pl${PL}_ol${OL}_bs4_cc1_tp1_seedNA__${RUNID} \
  #   python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-1p3b 4 "${PL}" ${OL} --mode vllm --tp 1 --num-iters 3 --num-warmup 3 \
  #   --model-name ${MODEL_PATH} --gpu-memory-utilization 0.85 --swap-space 8 --max-paddings 256
}

if [ "$1" = "profile_smoke" ]; then
  profile_smoke
fi
