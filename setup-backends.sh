#!/usr/bin/env bash

HOST="$(hostname -s)"
export HOST
export MACHINE="$HOST"
export EXCL_HOST="$HOST"

echo "Setting up backends for $HOST"

strip_ld_library_path_entry() {
  local remove_dir="$1"
  local old="${LD_LIBRARY_PATH:-}"
  local new=""
  local IFS=':'
  local part

  for part in $old; do
    [ -n "$part" ] || continue
    [ "$part" != "$remove_dir" ] || continue
    if [ -z "$new" ]; then
      new="$part"
    else
      new="$new:$part"
    fi
  done

  export LD_LIBRARY_PATH="$new"
}

append_ld_library_path() {
  local dir="$1"
  [ -n "$dir" ] || return 0
  [ -d "$dir" ] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$dir" ;;
  esac
}

if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
  elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
  fi
fi

have_module_loaded() {
  module list 2>&1 | grep -q "$1"
}

load_module_if_needed() {
  local mod="$1"
  if ! have_module_loaded "$mod"; then
    module load "$mod"
  fi
}

# Repo root = directory containing this script
THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOP_LEVEL="${TOP_LEVEL:-$THIS_DIR}"
export TOP_LEVEL

# Single canonical install root:
#   $REPO/sycl/sycl-implementations/<host>
SYCL_ROOT="${SYCL_ROOT:-$TOP_LEVEL/sycl}"
IMPL_BASE="${IMPL_BASE:-$SYCL_ROOT/sycl-implementations}"

export SYCL_ROOT
export IMPL_BASE

export BACKENDS="${BACKENDS:-}"
export CUDA_DEV_TARGET="${CUDA_DEV_TARGET:-}"
export HIP_DEV_TARGET="${HIP_DEV_TARGET:-}"

export ONEAPI_LIB_DEP="${ONEAPI_LIB_DEP:-}"
export ONEAPI_INSTALL_DIR="${ONEAPI_INSTALL_DIR:-}"
export ICPX="${ICPX:-}"

export UNISYCL_LLVM_PREFIX="${UNISYCL_LLVM_PREFIX:-}"
export UNISYCL_LLVM_PATH="${UNISYCL_LLVM_PATH:-}"
export UNISYCL_LLVM_DIR="${UNISYCL_LLVM_DIR:-}"
export UNISYCL_CLANG_DIR="${UNISYCL_CLANG_DIR:-}"
export UNISYCL_LLVM_LD="${UNISYCL_LLVM_LD:-}"

export SOURCE_LLVM_VERSION="${SOURCE_LLVM_VERSION:-19.0.1}"
export SOURCE_LLVM_MAJOR="${SOURCE_LLVM_MAJOR:-19}"
export SOURCE_LLVM_INSTALL_ROOT="${SOURCE_LLVM_INSTALL_ROOT:-$IMPL_BASE/$HOST/llvm-${SOURCE_LLVM_VERSION}}"

export DPCPPCPU_INSTALL_ROOT="${DPCPPCPU_INSTALL_ROOT:-$IMPL_BASE/$HOST/dpc++-cpu}"
export DPCPPCPU="${DPCPPCPU:-$DPCPPCPU_INSTALL_ROOT/bin/clang++}"

export DPCPPCUDA_INSTALL_ROOT="${DPCPPCUDA_INSTALL_ROOT:-$IMPL_BASE/$HOST/dpc++-cuda}"
export DPCPPHIP_INSTALL_ROOT="${DPCPPHIP_INSTALL_ROOT:-$IMPL_BASE/$HOST/dpc++-hip}"
export ADAPTIVECPP_INSTALL_ROOT="${ADAPTIVECPP_INSTALL_ROOT:-$IMPL_BASE/$HOST/adaptivecpp}"
export IRIS_INSTALL_ROOT="${IRIS_INSTALL_ROOT:-$IMPL_BASE/$HOST/iris}"
export UNISYCL_INSTALL_ROOT="${UNISYCL_INSTALL_ROOT:-$IMPL_BASE/$HOST/unisycl}"

export DPCPP_CUDA_PATH="${DPCPP_CUDA_PATH:-}"

case "$HOST" in
  milan2)
    export BACKENDS="cuda,openmp"
    export CUDA_DEV_TARGET="sm_70"
    export ONEAPI_LIB_DEP="/opt/intel/oneapi/2025.0/lib:/opt/intel/oneapi/compiler/2025.0/lib"
    append_ld_library_path "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib"
    load_module_if_needed "llvm/19.0.1_rc3_70_80_90-offload"
    ;;

  milan0)
    export BACKENDS="cuda,openmp"
    export CUDA_DEV_TARGET="sm_80"
    export ONEAPI_LIB_DEP="/opt/intel/oneapi/2025.0/lib:/opt/intel/oneapi/compiler/2025.0/lib"
    append_ld_library_path "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib"
    load_module_if_needed "llvm/19.0.1_rc3_70_80_90-offload"
    load_module_if_needed "gcc/12.1.0"
    ;;

  faraday)
    export BACKENDS="openmp,hip"
    export HIP_DEV_TARGET="gfx942"
    export ONEAPI_LIB_DEP="/opt/intel/oneapi/2025.2/lib"
    append_ld_library_path "/opt/rocm/llvm/lib"
    load_module_if_needed "cmake/4.1.1"
    load_module_if_needed "gcc/13.2.0"
    load_module_if_needed "llvm/19.0.1_rc3_70_80_90-offload"
    ;;

  cousteau)
    export BACKENDS="openmp,hip"
    export HIP_DEV_TARGET="gfx908"
    export ONEAPI_LIB_DEP="/opt/intel/oneapi/2025.2/lib"
    append_ld_library_path "/opt/rocm/llvm/lib"
    load_module_if_needed "gcc/13.2.0"
    load_module_if_needed "llvm/19.0.1_rc3_70_80_90-offload"
    ;;

  zenith)
    export BACKENDS="cuda,hip,openmp"
    export CUDA_DEV_TARGET="sm_86"
    export HIP_DEV_TARGET="gfx1030"
    export CUDA_PATH="${CUDA_PATH:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9}"
    export DPCPP_CUDA_PATH="${DPCPP_CUDA_PATH:-$CUDA_PATH}"
    export DPCPP_CUPTI_INCLUDE_DIR="${DPCPP_CUPTI_INCLUDE_DIR:-$CUDA_PATH/extras/CUPTI/include}"
    export DPCPP_CUPTI_LIBRARY="${DPCPP_CUPTI_LIBRARY:-$CUDA_PATH/extras/CUPTI/lib64/libcupti.so}"
    append_ld_library_path "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/REDIST/cuda/12.9/extras/CUPTI/lib64"
    load_module_if_needed "cmake/4.1.1"
    load_module_if_needed "gcc/13.2.0"
    load_module_if_needed "llvm/19.0.1_rc3_70_80_90-offload"
    ;;

  hudson)
    export BACKENDS="cuda,openmp"
    export CUDA_DEV_TARGET="sm_90"
    export CUDA_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/12.6"
    export NCORES="${NCORES:-64}"
    ;;

  equinox)
    export BACKENDS="cuda,openmp"
    export CUDA_DEV_TARGET="sm_70"
    ;;

  leconte)
    export BACKENDS="cuda,openmp"
    export CUDA_DEV_TARGET="sm_70"
    ;;

  oswald00)
    export BACKENDS="cuda,openmp"
    export CUDA_DEV_TARGET="sm_60"
    ;;

  explorer)
    export BACKENDS="hip,openmp"
    export HIP_DEV_TARGET="gfx906"
    load_module_if_needed "llvm/17.0.6-SM_70-offload"
    ;;

  radeon)
    export BACKENDS="hip,openmp"
    export HIP_DEV_TARGET="gfx906"
    load_module_if_needed "cmake/4.1.1"
    ;;

  clark)
    export BACKENDS="openmp"
    ;;

  pharaoh)
    export BACKENDS="openmp"
    ;;

  *)
    if [ "${AURORA_HOST:-0}" = "1" ]; then
      export BACKENDS="openmp,levelzero"
      load_module_if_needed "gcc/13.3.0"
    else
      echo "Unknown system ($HOST)." >&2
      exit 1
    fi
    ;;
esac

if ! have_module_loaded "cmake"; then
  load_module_if_needed "cmake/4.1.1"
fi

if ! have_module_loaded "gcc"; then
  load_module_if_needed "gcc/13.2.0"
fi

if command -v clang >/dev/null 2>&1; then
  CLANG_BIN="$(command -v clang)"
  DETECTED_LLVM_PREFIX="$(dirname "$(dirname "$CLANG_BIN")")"

  export UNISYCL_LLVM_PREFIX="${UNISYCL_LLVM_PREFIX:-$DETECTED_LLVM_PREFIX}"
  export UNISYCL_LLVM_PATH="${UNISYCL_LLVM_PATH:-$UNISYCL_LLVM_PREFIX/bin}"
  export UNISYCL_LLVM_DIR="${UNISYCL_LLVM_DIR:-$UNISYCL_LLVM_PREFIX/lib/cmake/llvm}"
  export UNISYCL_CLANG_DIR="${UNISYCL_CLANG_DIR:-$UNISYCL_LLVM_PREFIX/lib/cmake/clang}"
  export UNISYCL_LLVM_LD="${UNISYCL_LLVM_LD:-$UNISYCL_LLVM_PREFIX/bin/ld.lld}"
fi

if [[ -n "${UNISYCL_LLVM_PREFIX:-}" ]]; then
  append_ld_library_path "$UNISYCL_LLVM_PREFIX/lib"
  append_ld_library_path "$UNISYCL_LLVM_PREFIX/lib/x86_64-unknown-linux-gnu"
fi

if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
  if [[ -z "${CUDA_PATH:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi

  export CUDA_TOOLKIT_ROOT_PATH="${CUDA_TOOLKIT_ROOT_PATH:-$CUDA_PATH}"
  export CUDA_LIB_PATH="${CUDA_LIB_PATH:-${CUDA_PATH}/targets/x86_64-linux/lib/stubs}"

  if [[ -n "${CUDA_TOOLKIT_ROOT_PATH:-}" ]]; then
    export NVCXX="${NVCXX:-${CUDA_TOOLKIT_ROOT_PATH}/bin/nvc++}"
    export NVMATHLIBS="${NVMATHLIBS:-$(dirname "${CUDA_TOOLKIT_ROOT_PATH}")/math_libs}"
  fi
fi

if [[ "${BACKENDS:-}" == *"hip"* ]]; then
  if [[ -z "${ROCM_PATH:-}" ]] && command -v hipcc >/dev/null 2>&1; then
    export ROCM_PATH="$(dirname "$(dirname "$(command -v hipcc)")")"
  fi
fi

if [[ "${BACKENDS:-}" == *"levelzero"* ]]; then
  export ONEAPI_INSTALL_DIR="${ONEAPI_INSTALL_DIR:-/opt/intel/oneapi/2025.0}"
  export ONEAPI_LIB_DEP="${ONEAPI_LIB_DEP:-$ONEAPI_INSTALL_DIR/lib}"
  export ICPX="${ICPX:-$ONEAPI_INSTALL_DIR/bin/icpx}"
fi

strip_ld_library_path_entry "$HOME/.iris/lib"
strip_ld_library_path_entry "$HOME/.iris/lib64"
append_ld_library_path "$IRIS_INSTALL_ROOT/lib"

export GCC_BIN="${GCC_BIN:-$(command -v gcc)}"
export GXX_BIN="${GXX_BIN:-$(command -v g++)}"
export GCC_DIR="${GCC_DIR:-$(dirname "$(dirname "$GCC_BIN")")}"
append_ld_library_path "$GCC_DIR/lib64"
append_ld_library_path "$GCC_DIR/lib"

export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"

export DPCPPHIP="${DPCPPHIP:-$DPCPPHIP_INSTALL_ROOT/bin/clang++}"
export DPCPPCUDA="${DPCPPCUDA:-$DPCPPCUDA_INSTALL_ROOT/bin/clang++}"

if [ -f "$DPCPPCUDA" ]; then
  export DPCPP="$DPCPPCUDA"
elif [ -f "$DPCPPHIP" ]; then
  export DPCPP="$DPCPPHIP"
fi

export SOURCE_LLVM_BIN="$SOURCE_LLVM_INSTALL_ROOT/bin"
export SOURCE_LLVM_DIR="$SOURCE_LLVM_INSTALL_ROOT/lib/cmake/llvm"
export SOURCE_CLANG_DIR="$SOURCE_LLVM_INSTALL_ROOT/lib/cmake/clang"
export SOURCE_LLVM_LIB="$SOURCE_LLVM_INSTALL_ROOT/lib/libLLVM.so"
export SOURCE_LLVM_CLANG="$SOURCE_LLVM_INSTALL_ROOT/bin/clang"
export SOURCE_LLVM_CLANGXX="$SOURCE_LLVM_INSTALL_ROOT/bin/clang++"
export SOURCE_LLVM_LD="$SOURCE_LLVM_INSTALL_ROOT/bin/ld.lld"

# DPC++ runtime libraries
if [[ "${BACKENDS:-}" == *"openmp"* ]]; then
  append_ld_library_path "$DPCPPCPU_INSTALL_ROOT/lib"
fi

if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
  append_ld_library_path "$DPCPPCUDA_INSTALL_ROOT/lib"
fi

if [[ "${BACKENDS:-}" == *"hip"* ]]; then
  append_ld_library_path "$DPCPPHIP_INSTALL_ROOT/lib"
fi
