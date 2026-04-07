#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ARCHIVED_RESULTS_ROOT="${ARCHIVED_RESULTS_ROOT:-$SCRIPT_DIR/archived-results}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results}"
IMAGE_DIR="${IMAGE_DIR:-$SCRIPT_DIR/images}"

mkdir -p "$IMAGE_DIR"

PIXIRUN="$SCRIPT_DIR/setup-pixi.sh run"
source "$SCRIPT_DIR/setup-backends.sh"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$SCRIPT_DIR/cpp:$SCRIPT_DIR/omp:$SCRIPT_DIR/sycl"
source "$SCRIPT_DIR/utils.sh"
setup_runtime_paths

bootstrap_results_dir() {
  if [[ ! -d "$RESULTS_ROOT" ]]; then
    if [[ ! -d "$ARCHIVED_RESULTS_ROOT" ]]; then
      echo "Missing both results/ and archived-results/" >&2
      exit 1
    fi
    echo "Bootstrapping results/ from archived-results/..."
    cp -a "$ARCHIVED_RESULTS_ROOT" "$RESULTS_ROOT"
  fi
}

bootstrap_results_dir

with_dpcpp_cpu_runtime() {
  ONEAPI_DEVICE_SELECTOR='native_cpu:cpu' \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$DPCPPCPU_INSTALL_ROOT/lib" \
  "$@"
}

FRS_CPU_GLOB="$RESULTS_ROOT/frs_scaling_cpu_results/*.csv"
FRS_CUDA_GLOB="$RESULTS_ROOT/frs_scaling_cuda_results/*.csv"
FRS_HIP_GLOB="$RESULTS_ROOT/frs_scaling_hip_results/*.csv"
FRS_XPU_GLOB="$RESULTS_ROOT/frs_scaling_xpu_results/*.csv"

STACK_CPU_GLOB="$RESULTS_ROOT/scaling-results-cpu/*.csv"
STACK_GPU_GLOB="$RESULTS_ROOT/scaling-results-gpu/*.csv"

run_cpu_frs() {
  local outdir="$RESULTS_ROOT/frs_scaling_cpu_results"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  local GRID_SIZE="${GRID_SIZE:-100}"
  local N_TRIALS="${N_TRIALS:-50}"

  cd "$SCRIPT_DIR"

  for i in $(seq 4 7); do
    local N_GEN_EVENTS=$((10 ** i))
    echo "Running CPU fixed-resource scaling for ${N_GEN_EVENTS} events..."

    $PIXIRUN python3 ./pytorch_2dloits.py \
      --torch_device="cpu" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-cpu_${N_GEN_EVENTS}.csv"

    $PIXIRUN python3 ./pytorch_2dloits.py \
      --cpp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/cpp_${N_GEN_EVENTS}.csv"

    $PIXIRUN python3 ./pytorch_2dloits.py \
      --omp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/omp_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=omp \
    $PIXIRUN python3 ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp-omp_${N_GEN_EVENTS}.csv"

    #export ONEAPI_DEVICE_SELECTOR='native_cpu:cpu'
    #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./sycl/sycl-implementations/milan2/dpc++-cpu/lib
    with_dpcpp_cpu_runtime \
      $PIXIRUN python3 ./pytorch_2dloits.py \
        --sycl_sampler \
        --sycl_implementations="dpcp" \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS"
    mv times.csv "$outdir/dpcp-tbb_${N_GEN_EVENTS}.csv"
    #unset ONEAPI_DEVICE_SELECTOR
  done
}

run_cpu_stacked() {
  local outdir="$RESULTS_ROOT/scaling-results-cpu"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  local GRID_SIZE="${GRID_SIZE:-100}"
  local N_TRIALS="${N_TRIALS:-10}"

  cd "$SCRIPT_DIR"

  for i in $(seq 4 7); do
    local N_GEN_EVENTS=$((10 ** i))
    echo "Profiling CPU implementations on ${N_GEN_EVENTS} events..."

    python3 ./pytorch_2dloits.py \
      --cpp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/cpp_${N_GEN_EVENTS}.csv"

    python3 ./pytorch_2dloits.py \
      --omp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/omp_${N_GEN_EVENTS}.csv"

    python3 ./pytorch_2dloits.py \
      --torch_device="cpu" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/pytorch-omp_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=omp \
    python3 ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/acpp-omp_${N_GEN_EVENTS}.csv"

    with_dpcpp_cpu_runtime \
      python3 ./pytorch_2dloits.py \
        --sycl_sampler \
        --sycl_implementations="dpcp" \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS" \
        --internal_timing="$outdir/dpcp-tbb_${N_GEN_EVENTS}.csv"
  done
}

run_cuda_frs() {
  local outdir="$RESULTS_ROOT/frs_scaling_cuda_results"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  local GRID_SIZE="${GRID_SIZE:-100}"
  local N_TRIALS="${N_TRIALS:-50}"

  cd "$SCRIPT_DIR"

  for i in $(seq 4 7); do
    local N_GEN_EVENTS=$((10 ** i))
    echo "Running CUDA fixed-resource scaling for ${N_GEN_EVENTS} events..."

    CUDA_VISIBLE_DEVICES=0 \
    python3 ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-cuda_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=cuda \
    python3 ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp-cuda_${N_GEN_EVENTS}.csv"

    python3 ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="dpcp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/dpcp-cuda_${N_GEN_EVENTS}.csv"
  done
}

run_hip_frs() {
  local outdir="$RESULTS_ROOT/frs_scaling_hip_results"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  local GRID_SIZE="${GRID_SIZE:-100}"
  local N_TRIALS="${N_TRIALS:-50}"

  cd "$SCRIPT_DIR"

  for i in $(seq 4 7); do
    local N_GEN_EVENTS=$((10 ** i))
    echo "Running HIP fixed-resource scaling for ${N_GEN_EVENTS} events..."

    HIP_VISIBLE_DEVICES=0 \
    python3 ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-hip_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=hip \
    python3 ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp-hip_${N_GEN_EVENTS}.csv"

    python3 ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="dpcp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/dpcp-hip_${N_GEN_EVENTS}.csv"
  done
}

plot_results() {
  if ! compgen -G "$FRS_CPU_GLOB" > /dev/null; then
    echo "No CPU fixed-resource CSV files found at: $FRS_CPU_GLOB" >&2
    exit 1
  fi

  export IMPLEMENTATION_SUBSTITUTIONS="cpp:C++,omp:OpenMP,pytorch-cpu:PyTorch (CPU),dpcp-tbb:SYCL DPC++ (TBB),acpp-omp:SYCL AdaptiveCpp (OpenMP),dpcp-cuda:SYCL DPC++ (CUDA),acpp-cuda:SYCL AdaptiveCpp (CUDA),pytorch-cuda:PyTorch (CUDA),dpcp-hip:SYCL DPC++ (HIP),acpp-hip:SYCL AdaptiveCpp (HIP),pytorch-hip:PyTorch (HIP),dpcp-xpu:SYCL DPC++ (XPU),pytorch-xpu:PyTorch (XPU)"

  local all_frs_paths="$FRS_CPU_GLOB:$FRS_CUDA_GLOB:$FRS_XPU_GLOB:$FRS_HIP_GLOB"

  TITLE="Fixed-Resource Scaling" \
  RESULT_PATH="$all_frs_paths" \
  FIGURE_PATH="$IMAGE_DIR/cpu_scaling.png" \
  ./utils/plot_frs_scaling.py \
    "PyTorch (CPU)" \
    "C++" \
    "OpenMP" \
    "SYCL AdaptiveCpp (OpenMP)" \
    "SYCL DPC++ (TBB)"

  TITLE="Fixed-Resource Scaling" \
  RESULT_PATH="$all_frs_paths" \
  FIGURE_PATH="$IMAGE_DIR/gpu_scaling.png" \
  ./utils/plot_frs_scaling.py \
    "PyTorch (CUDA)" \
    "SYCL AdaptiveCpp (CUDA)" \
    "SYCL DPC++ (CUDA)" \
    "PyTorch (HIP)" \
    "SYCL AdaptiveCpp (HIP)" \
    "SYCL DPC++ (HIP)" \
    "PyTorch (XPU)" \
    "SYCL DPC++ (XPU)"

  if compgen -G "$STACK_CPU_GLOB" > /dev/null; then
    TITLE="Scaling and Breakdown of 2D-LOITS" \
    RESULT_PATH="$STACK_CPU_GLOB" \
    FIGURE_PATH="$IMAGE_DIR/stacked_barplot_cpu.pdf" \
    SCALE_AXES=1 \
    DROP_LEGEND=1 \
    ./utils/plot_stacked_boxplots.py \
      "PyTorch (CPU)" \
      "SYCL AdaptiveCpp (OpenMP)" \
      "SYCL DPC++ (TBB)" \
      "C++" \
      "OpenMP"
  fi

  if compgen -G "$STACK_GPU_GLOB" > /dev/null; then
    TITLE="Scaling and Breakdown of 2D-LOITS" \
    RESULT_PATH="$STACK_GPU_GLOB" \
    FIGURE_PATH="$IMAGE_DIR/stacked_barplot_gpu.pdf" \
    SCALE_AXES=1 \
    DROP_Y_AX=1 \
    ./utils/plot_stacked_boxplots.py \
      "PyTorch (CUDA)" \
      "PyTorch (HIP)" \
      "PyTorch (XPU)" \
      "SYCL AdaptiveCpp (CUDA)" \
      "SYCL AdaptiveCpp (HIP)" \
      "SYCL DPC++ (CUDA)" \
      "SYCL DPC++ (HIP)" \
      "SYCL DPC++ (XPU)"
  fi

  echo "Wrote $IMAGE_DIR/cpu_scaling.png"
  echo "Wrote $IMAGE_DIR/gpu_scaling.png"
  if [[ -f "$IMAGE_DIR/stacked_barplot_cpu.pdf" ]]; then
    echo "Wrote $IMAGE_DIR/stacked_barplot_cpu.pdf"
  fi
  if [[ -f "$IMAGE_DIR/stacked_barplot_gpu.pdf" ]]; then
    echo "Wrote $IMAGE_DIR/stacked_barplot_gpu.pdf"
  fi
}

if [[ -n "${DORUN:-}" ]]; then
  # CPU phase: make sure canonical DPC++ wrapper is the TBB one
  $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_tbb CXX=g++ CC=gcc
  run_cpu_frs
  run_cpu_stacked

  if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
    # CUDA phase: switch canonical DPC++ wrapper to CUDA
    $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_cuda CXX=g++ CC=gcc
    run_cuda_frs
  fi

  if [[ "${BACKENDS:-}" == *"hip"* ]]; then
    # HIP phase: switch canonical DPC++ wrapper to HIP
    $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_hip CXX=g++ CC=gcc
    run_hip_frs
  fi

  if [[ "${BACKENDS:-}" == *"levelzero"* || "${BACKENDS:-}" == *"xpu"* ]]; then
    # XPU phase: switch canonical DPC++ wrapper to SPIR-V/XPU
    $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_spirv CXX=g++ CC=gcc
    run_xpu_frs
  fi
fi 
plot_results
