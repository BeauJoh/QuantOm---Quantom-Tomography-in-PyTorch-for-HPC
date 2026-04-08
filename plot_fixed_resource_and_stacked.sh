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
  $PIXIRUN python "$@"
}

with_dpcpp_cuda_runtime() {
  ONEAPI_DEVICE_SELECTOR='cuda:gpu' \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$DPCPPCUDA_INSTALL_ROOT/lib" \
  $PIXIRUN python "$@"
}

with_dpcpp_hip_runtime() {
  ONEAPI_DEVICE_SELECTOR='hip:gpu' \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$DPCPPHIP_INSTALL_ROOT/lib" \
  $PIXIRUN python "$@"
}

FRS_CPU_GLOB="$RESULTS_ROOT/frs_scaling_cpu_results/*.csv"
FRS_CUDA_GLOB="$RESULTS_ROOT/frs_scaling_cuda_results/*.csv"
FRS_HIP_GLOB="$RESULTS_ROOT/frs_scaling_hip_results/*.csv"
FRS_XPU_GLOB="$RESULTS_ROOT/frs_scaling_xpu_results/*.csv"

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

    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="cpu" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-cpu_${N_GEN_EVENTS}.csv"

    $PIXIRUN python ./pytorch_2dloits.py \
      --cpp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/cpp_${N_GEN_EVENTS}.csv"

    $PIXIRUN python ./pytorch_2dloits.py \
      --omp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/omp_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=omp \
    $PIXIRUN python ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp-omp_${N_GEN_EVENTS}.csv"

    with_dpcpp_cpu_runtime ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="dpcp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/dpcp-tbb_${N_GEN_EVENTS}.csv"
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

    $PIXIRUN python ./pytorch_2dloits.py \
      --cpp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/cpp_${N_GEN_EVENTS}.csv"

    $PIXIRUN python ./pytorch_2dloits.py \
      --omp_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/omp_${N_GEN_EVENTS}.csv"

    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="cpu" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/pytorch-omp_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=omp \
    $PIXIRUN python ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS" \
      --internal_timing="$outdir/acpp-omp_${N_GEN_EVENTS}.csv"

    with_dpcpp_cpu_runtime ./pytorch_2dloits.py \
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
    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-cuda_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=cuda \
    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp-cuda_${N_GEN_EVENTS}.csv"

    with_dpcpp_cuda_runtime ./pytorch_2dloits.py \
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
    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-hip_${N_GEN_EVENTS}.csv"

    ACPP_VISIBILITY_MASK=hip \
    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="cuda:0" \
      --sycl_sampler \
      --sycl_implementations="acpp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp-hip_${N_GEN_EVENTS}.csv"

    with_dpcpp_hip_runtime ./pytorch_2dloits.py \
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

run_xpu_frs() {
  local outdir="$RESULTS_ROOT/frs_scaling_xpu_results"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  local GRID_SIZE="${GRID_SIZE:-100}"
  local N_TRIALS="${N_TRIALS:-50}"

  cd "$SCRIPT_DIR"

  for i in $(seq 4 7); do
    local N_GEN_EVENTS=$((10 ** i))
    echo "Running XPU fixed-resource scaling for ${N_GEN_EVENTS} events..."

    $PIXIRUN python ./pytorch_2dloits.py \
      --torch_device="xpu" \
      --pytorch_sampler \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch-xpu_${N_GEN_EVENTS}.csv"

    $PIXIRUN python ./pytorch_2dloits.py \
      --sycl_sampler \
      --sycl_implementations="dpcp" \
      --disable_gradient_tracking \
      --threading=False \
      --n_gen_events="$N_GEN_EVENTS" \
      --grid_size="$GRID_SIZE" \
      --n_trials="$N_TRIALS"
    mv times.csv "$outdir/dpcp-xpu_${N_GEN_EVENTS}.csv"
  done
}

plot_results() {
  if ! compgen -G "$FRS_CPU_GLOB" > /dev/null; then
    echo "No CPU fixed-resource CSV files found at: $FRS_CPU_GLOB" >&2
    exit 1
  fi

  $PIXIRUN python "$SCRIPT_DIR/plotscripts/plot_cpu.py" \
    --results-root "$RESULTS_ROOT" \
    --output "$IMAGE_DIR/cpu_scaling.png"

  $PIXIRUN python "$SCRIPT_DIR/plotscripts/plot_gpu.py" \
    --results-root "$RESULTS_ROOT" \
    --output "$IMAGE_DIR/gpu_scaling.png"

  echo "Wrote $IMAGE_DIR/cpu_scaling.png"
  echo "Wrote $IMAGE_DIR/gpu_scaling.png"
}

if [[ -n "${DORUN:-}" ]]; then
  $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_tbb CXX=g++ CC=gcc
  run_cpu_frs
  run_cpu_stacked

  if [[ "${FORCE_CUDA:-0}" == "1" || "${BACKENDS:-}" == *"cuda"* ]]; then
    $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_cuda CXX=g++ CC=gcc
    run_cuda_frs
  fi

  if [[ "${FORCE_HIP:-0}" == "1" || "${BACKENDS:-}" == *"hip"* ]]; then
    $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_hip CXX=g++ CC=gcc
    run_hip_frs
  fi

  if [[ "${FORCE_XPU:-0}" == "1" || "${BACKENDS:-}" == *"levelzero"* || "${BACKENDS:-}" == *"xpu"* ]]; then
    $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_spirv CXX=g++ CC=gcc
    run_xpu_frs
  fi
fi

plot_results
