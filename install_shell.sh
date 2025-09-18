# 1) Limit CMake/Ninja parallel build jobs
export CMAKE_BUILD_PARALLEL_LEVEL=8

# 2) Many projects also honor this (PyTorch-style)
export MAX_JOBS=8

# 3) If Ninja flags are picked up by the build wrapper
export NINJAFLAGS="-j8"

# 4) Profiler-friendly build config + restrict GPU archs to your GPU
#   86 = Ampere (RTX 30xx/A6000), 89 = Ada (RTX 40xx), 80 = A100, etc.
export TORCH_CUDA_ARCH_LIST="8.9"
# CURSOR: Review this line - Enabling host debug symbols (-g) in optimized builds for profiling
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DCMAKE_CUDA_FLAGS_RELWITHDEBINFO='-lineinfo -Xcompiler=-fno-omit-frame-pointer' -DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O2 -g -fno-omit-frame-pointer' -DCMAKE_C_FLAGS_RELWITHDEBINFO='-O2 -g -fno-omit-frame-pointer' -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# Re-run install with no deps and no build isolation to avoid extra envs
pip install -e /mnt/d/Corvex/Excercise/vllm --no-deps -v