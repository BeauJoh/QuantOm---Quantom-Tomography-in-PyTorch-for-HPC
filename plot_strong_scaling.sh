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

RESULT_GLOB="$RESULTS_ROOT/strong_scaling_results/*.csv"

run_results() {
  local outdir="$RESULTS_ROOT/strong_scaling_results"
  rm -rf "$outdir"
  mkdir -p "$outdir"

  local N_GEN_EVENTS="${N_GEN_EVENTS:-100000000}"
  local GRID_SIZE="${GRID_SIZE:-100}"
  local N_TRIALS="${N_TRIALS:-10}"

  cd "$SCRIPT_DIR"

  for i in $(seq 0 6); do
    local max_cpu_id=$((2 ** i - 1))
    local ncores=$((max_cpu_id + 1))

    echo "Running strong scaling at ${ncores} core(s)..."

    OMP_NUM_THREADS="$ncores" \
    MKL_NUM_THREADS="$ncores" \
    taskset -c 0-"$max_cpu_id" \
      $PIXIRUN python ./pytorch_2dloits.py \
        --torch_device="cpu" \
        --pytorch_sampler \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS"
    mv times.csv "$outdir/pytorch_${ncores}.csv"

    taskset -c 0-"$max_cpu_id" \
      $PIXIRUN python ./pytorch_2dloits.py \
        --cpp_sampler \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS"
    mv times.csv "$outdir/cpp_${ncores}.csv"

    OMP_NUM_THREADS="$ncores" \
    taskset -c 0-"$max_cpu_id" \
      $PIXIRUN python ./pytorch_2dloits.py \
        --omp_sampler \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS"
    mv times.csv "$outdir/omp_${ncores}.csv"

    ACPP_VISIBILITY_MASK=omp \
    OMP_NUM_THREADS="$ncores" \
    taskset -c 0-"$max_cpu_id" \
      $PIXIRUN python ./pytorch_2dloits.py \
        --sycl_sampler \
        --sycl_implementations="acpp" \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS"
    mv times.csv "$outdir/acpp_omp_${ncores}.csv"

    ONEAPI_DEVICE_SELECTOR='native_cpu:cpu' \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$DPCPPCPU_INSTALL_ROOT/lib" \
    taskset -c 0-"$max_cpu_id" \
      $PIXIRUN python ./pytorch_2dloits.py \
        --sycl_sampler \
        --sycl_implementations="dpcp" \
        --disable_gradient_tracking \
        --threading=False \
        --n_gen_events="$N_GEN_EVENTS" \
        --grid_size="$GRID_SIZE" \
        --n_trials="$N_TRIALS"
    mv times.csv "$outdir/dpcp_tbb_${ncores}.csv"
  done
}

plot_results() {
  if ! compgen -G "$RESULT_GLOB" > /dev/null; then
    echo "No strong scaling CSV files found at: $RESULT_GLOB" >&2
    exit 1
  fi

  $PIXIRUN python "$SCRIPT_DIR/plotscripts/plot_ss.py" \
    --results-root "$RESULTS_ROOT" \
    --output "$IMAGE_DIR/strong_scaling.png"

  echo "Wrote $IMAGE_DIR/strong_scaling.png"
}

if [[ -n "${DORUN:-}" ]]; then
  $PIXIRUN make -C "$SCRIPT_DIR/sycl" dpc++_for_tbb CXX=g++ CC=gcc
  run_results
fi

plot_results
