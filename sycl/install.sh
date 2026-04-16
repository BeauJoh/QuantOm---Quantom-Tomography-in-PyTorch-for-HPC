#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./install.sh [options]

Installs SYCL implementations into:
  ./sycl-implementations/<host>/

Default behavior:
  - installs everything that is missing
  - does not rebuild components that already exist
  - if REINSTALL=1 is set, removes the host install directory first

Options:
  -h, --help              Show this help message and exit
  --reinstall             Remove ./sycl-implementations/<host> first
  --unisycl-only          Build/install UniSYCL (+ IRIS if missing) only
  --llvm-only             Build/install source LLVM only
  --dpcpp-only            Build/install DPC++ only
  --acpp-only             Build/install AdaptiveCpp only

Environment overrides:
  REINSTALL=1             Same as --reinstall
  INSTALL_UNISYCL=0|1     Enable/disable UniSYCL install
  INSTALL_LLVM=0|1        Enable/disable source LLVM install
  INSTALL_DPCPP=0|1       Enable/disable DPC++ install
  INSTALL_ACPP=0|1        Enable/disable AdaptiveCpp install
  NCORES=<n>              Parallel build jobs
  GIT_SUBMODULE_JOBS=<n>  Parallel submodule jobs (default: 4)
EOF
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INSTALL_UNISYCL="${INSTALL_UNISYCL:-1}"
INSTALL_LLVM="${INSTALL_LLVM:-1}"
INSTALL_DPCPP="${INSTALL_DPCPP:-1}"
INSTALL_ACPP="${INSTALL_ACPP:-1}"
REINSTALL="${REINSTALL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --reinstall)
      REINSTALL=1
      shift
      ;;
    --unisycl-only)
      INSTALL_UNISYCL=1
      INSTALL_LLVM=0
      INSTALL_DPCPP=0
      INSTALL_ACPP=0
      shift
      ;;
    --llvm-only)
      INSTALL_UNISYCL=0
      INSTALL_LLVM=1
      INSTALL_DPCPP=0
      INSTALL_ACPP=0
      shift
      ;;
    --dpcpp-only)
      INSTALL_UNISYCL=0
      INSTALL_LLVM=1
      INSTALL_DPCPP=1
      INSTALL_ACPP=0
      shift
      ;;
    --acpp-only)
      INSTALL_UNISYCL=0
      INSTALL_LLVM=1
      INSTALL_DPCPP=0
      INSTALL_ACPP=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Try ./install.sh --help" >&2
      exit 2
      ;;
  esac
done

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../setup-backends.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../utils.sh"

TOP_LEVEL="${TOP_LEVEL:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
HOST="${HOST:-$(hostname -s)}"
IMPL_DIR="${IMPL_DIR:-$SCRIPT_DIR/sycl-implementations/$HOST}"
SOURCES_DIR="${SOURCES_DIR:-$SCRIPT_DIR/sycl-implementations/sources}"
IRIS_INSTALL_ROOT="${IRIS_INSTALL_ROOT:-$IMPL_DIR/iris}"
UNISYCL_INSTALL_ROOT="${UNISYCL_INSTALL_ROOT:-$IMPL_DIR/unisycl}"

export TOP_LEVEL HOST IMPL_DIR SOURCES_DIR IRIS_INSTALL_ROOT UNISYCL_INSTALL_ROOT

mkdir -p "$IMPL_DIR" "$SOURCES_DIR"

if [[ -n "$REINSTALL" ]]; then
  warning "REINSTALL requested: removing $IMPL_DIR"
  rm -rf "$IMPL_DIR"
  mkdir -p "$IMPL_DIR"
fi

export PYTHON="${PYTHON:-$(command -v python3)}"
export NCORES="${NCORES:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || awk '/^processor/{n+=1}END{print n}' /proc/cpuinfo)}"

experiment "Install configuration"
info "HOST=$HOST"
info "TOP_LEVEL=$TOP_LEVEL"
info "IMPL_DIR=$IMPL_DIR"
info "SOURCES_DIR=$SOURCES_DIR"
info "IRIS_INSTALL_ROOT=$IRIS_INSTALL_ROOT"
info "UNISYCL_INSTALL_ROOT=$UNISYCL_INSTALL_ROOT"
info "BACKENDS=${BACKENDS:-}"
info "NCORES=$NCORES"
info "INSTALL_UNISYCL=$INSTALL_UNISYCL"
info "INSTALL_LLVM=$INSTALL_LLVM"
info "INSTALL_DPCPP=$INSTALL_DPCPP"
info "INSTALL_ACPP=$INSTALL_ACPP"
info "UNISYCL_LLVM_PREFIX=${UNISYCL_LLVM_PREFIX:-}"
info "SOURCE_LLVM_INSTALL_ROOT=${SOURCE_LLVM_INSTALL_ROOT:-}"

retry() {
  local attempts="$1"
  local delay="$2"
  shift 2

  local n=1
  until "$@"; do
    if [[ "$n" -ge "$attempts" ]]; then
      error "Command failed after $attempts attempts: $*"
      return 1
    fi
    warning "Command failed (attempt $n/$attempts): $*"
    warning "Retrying in ${delay}s..."
    sleep "$delay"
    n=$((n + 1))
  done
}

ensure_rust() {
  if [[ ! -d "$HOME/.cargo" ]]; then
    experiment "Installing Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
}

prepare_unisycl_source() {
  experiment "Preparing UniSYCL source"
  cd "$SOURCES_DIR"

  if [[ ! -d "$SOURCES_DIR/unisycl-src" ]]; then
    retry 3 10 git clone -b sycl-bench-changes git@code.ornl.gov:fujita/charm-sycl unisycl-src
  fi

  cd "$SOURCES_DIR/unisycl-src"

  git config --local url."https://".insteadOf git://
  git config --local http.postBuffer 52428800000
  git config --local fetch.recurseSubmodules false

  retry 5 15 git submodule sync --recursive
  retry 5 15 git submodule update --init --recursive --jobs "${GIT_SUBMODULE_JOBS:-4}"
}

install_iris_if_missing() {
  if [[ -d "$IRIS_INSTALL_ROOT" ]]; then
    info "IRIS already installed: $IRIS_INSTALL_ROOT"
    return
  fi

  experiment "Building and installing IRIS"
  cd "$SOURCES_DIR"
  rm -rf build-iris
  mkdir -p build-iris
  cd build-iris

  if [[ ! -d iris-src ]]; then
    git clone -b v3.0.0 git@github.com:ORNL/iris.git iris-src
  fi

  cd iris-src
  ./build.sh
}

install_unisycl() {
  [[ "$INSTALL_UNISYCL" == "1" ]] || return 0

  prepare_unisycl_source
  install_iris_if_missing
  ensure_rust

  if [[ -d "$UNISYCL_INSTALL_ROOT" ]]; then
    info "UniSYCL already installed: $UNISYCL_INSTALL_ROOT"
    return
  fi

  experiment "Building and installing UniSYCL"
  cd "$SOURCES_DIR/unisycl-src"

  : "${UNISYCL_LLVM_PREFIX:?UNISYCL_LLVM_PREFIX is not set; check setup-backends.sh}"
  : "${UNISYCL_LLVM_DIR:?UNISYCL_LLVM_DIR is not set; check setup-backends.sh}"
  : "${UNISYCL_CLANG_DIR:?UNISYCL_CLANG_DIR is not set; check setup-backends.sh}"
  : "${UNISYCL_LLVM_LD:?UNISYCL_LLVM_LD is not set; check setup-backends.sh}"

  local usy_cc="$UNISYCL_LLVM_PREFIX/bin/clang"
  local usy_cxx="$UNISYCL_LLVM_PREFIX/bin/clang++"

  [[ -x "$usy_cc" ]] || { error "UniSYCL clang not executable: $usy_cc"; exit 1; }
  [[ -x "$usy_cxx" ]] || { error "UniSYCL clang++ not executable: $usy_cxx"; exit 1; }
  [[ -x "$UNISYCL_LLVM_LD" ]] || { error "UniSYCL lld not executable: $UNISYCL_LLVM_LD"; exit 1; }
  [[ -f "$UNISYCL_LLVM_DIR/LLVMConfig.cmake" ]] || { error "Missing UniSYCL LLVMConfig.cmake"; exit 1; }
  [[ -f "$UNISYCL_CLANG_DIR/ClangConfig.cmake" ]] || { error "Missing UniSYCL ClangConfig.cmake"; exit 1; }

  unset IRIS_ROOT IRIS_DIR IRIS_INCLUDE_DIR IRIS_LIBRARY IRIS_INCLUDEDIR IRIS_LIBDIR

  export IRIS_DIR="$IRIS_INSTALL_ROOT"
  export IRIS_ROOT="$IRIS_INSTALL_ROOT"
  export IRIS_INCLUDE_DIR="$IRIS_INSTALL_ROOT/include"
  export IRIS_LIBRARY="$IRIS_INSTALL_ROOT/lib/libiris.so"

  [[ -f "$IRIS_INCLUDE_DIR/iris/iris.h" ]] || { error "Missing IRIS header"; exit 1; }
  [[ -f "$IRIS_LIBRARY" ]] || { error "Missing IRIS library"; exit 1; }

  experiment "Using the following for UniSYCL"
  info "CC=$usy_cc"
  info "CXX=$usy_cxx"
  info "UNISYCL_LLVM_PREFIX=$UNISYCL_LLVM_PREFIX"
  info "UNISYCL_LLVM_DIR=$UNISYCL_LLVM_DIR"
  info "UNISYCL_CLANG_DIR=$UNISYCL_CLANG_DIR"
  info "UNISYCL_LLVM_LD=$UNISYCL_LLVM_LD"
  info "IRIS_DIR=$IRIS_DIR"

  rm -rf build
  mkdir -p build
  cd build

  cmake .. \
    -G Ninja \
    -DCMAKE_GENERATOR:INTERNAL=Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$usy_cc" \
    -DCMAKE_CXX_COMPILER="$usy_cxx" \
    -DUSE_IRIS=TRUE \
    -DCHARM_SYCL_IRIS_IS_REQUIRED=TRUE \
    -DIRIS_DIR="$IRIS_INSTALL_ROOT" \
    -DIRIS_ROOT="$IRIS_INSTALL_ROOT" \
    -DIRIS_INCLUDE_DIR="$IRIS_INSTALL_ROOT/include" \
    -DIRIS_LIBRARY="$IRIS_INSTALL_ROOT/lib/libiris.so" \
    -DCMAKE_INSTALL_PREFIX="$UNISYCL_INSTALL_ROOT" \
    -DCMAKE_CXX_FLAGS="-std=c++14" \
    -DCHARM_SYCL_USE_CLANG_DYLIB=NO \
    -DCMAKE_POSITION_INDEPENDENT_CODE=YES \
    -DCSCC_USE_LINKER="$UNISYCL_LLVM_LD" \
    -DLLVM_DIR="$UNISYCL_LLVM_DIR" \
    -DClang_DIR="$UNISYCL_CLANG_DIR" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

  ninja
  ninja install
}

install_source_llvm() {
  [[ "$INSTALL_LLVM" == "1" ]] || return 0

  if [[ -d "$SOURCE_LLVM_INSTALL_ROOT" && -x "$SOURCE_LLVM_CLANG" && -f "$SOURCE_LLVM_LIB" ]]; then
    info "Source LLVM already installed: $SOURCE_LLVM_INSTALL_ROOT"
    return
  fi

  experiment "Building and installing source LLVM"
  cd "$SYCL_ROOT"
  "$SCRIPT_DIR/install-llvm.sh"
}

install_tbb_if_missing() {
  local tbb_src_dir="$SOURCES_DIR/tbb-src"
  local tbb_install_dir="$IMPL_DIR/tbb"

  if [[ -d "$tbb_install_dir" && -f "$tbb_install_dir/lib/cmake/TBB/TBBConfig.cmake" ]]; then
    info "oneTBB already installed: $tbb_install_dir"
    return 0
  fi

  experiment "Building and installing oneTBB"
  mkdir -p "$SOURCES_DIR"

  if [[ ! -d "$tbb_src_dir" ]]; then
    git clone https://github.com/oneapi-src/oneTBB.git "$tbb_src_dir"
  fi

  : "${GCC_BIN:?GCC_BIN is not set; check setup-backends.sh}"
  : "${GXX_BIN:?GXX_BIN is not set; check setup-backends.sh}"

  [[ -x "$GCC_BIN" ]] || { error "GCC_BIN not executable: $GCC_BIN"; exit 1; }
  [[ -x "$GXX_BIN" ]] || { error "GXX_BIN not executable: $GXX_BIN"; exit 1; }

  cd "$tbb_src_dir"
  rm -rf build
  mkdir -p build
  cd build

  experiment "Using the following for oneTBB"
  info "CC=$GCC_BIN"
  info "CXX=$GXX_BIN"

  cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$GCC_BIN" \
    -DCMAKE_CXX_COMPILER="$GXX_BIN" \
    -DCMAKE_INSTALL_PREFIX="$tbb_install_dir" \
    -DTBB_TEST=OFF

  ninja -j "${NCORES:-32}"
  ninja install

  [[ -f "$tbb_install_dir/lib/cmake/TBB/TBBConfig.cmake" ]] || {
    error "TBBConfig.cmake missing after oneTBB install"
    exit 1
  }
}

install_dpcpp() {
  [[ "$INSTALL_DPCPP" == "1" ]] || return 0

  install_source_llvm
  install_tbb_if_missing

  local need_cpu=0
  local need_cuda=0
  local need_hip=0

  [[ "${BACKENDS:-}" == *"openmp"* ]] && [[ ! -d "$IMPL_DIR/dpc++-cpu" ]] && need_cpu=1
  [[ "${BACKENDS:-}" == *"cuda"*   ]] && [[ ! -d "$IMPL_DIR/dpc++-cuda" ]] && need_cuda=1
  [[ "${BACKENDS:-}" == *"hip"*    ]] && [[ ! -d "$IMPL_DIR/dpc++-hip"  ]] && need_hip=1

  if [[ "$need_cpu" == "0" && "$need_cuda" == "0" && "$need_hip" == "0" ]]; then
    info "Requested DPC++ installs already exist under $IMPL_DIR"
    return 0
  fi

  experiment "Building and installing DPC++"
  cd "$SYCL_ROOT"

  [[ -x "$SOURCE_LLVM_CLANG" ]] || { error "DPC++ requires source-built LLVM clang: $SOURCE_LLVM_CLANG"; exit 1; }
  [[ -x "$SOURCE_LLVM_CLANGXX" ]] || { error "DPC++ requires source-built LLVM clang++: $SOURCE_LLVM_CLANGXX"; exit 1; }
  [[ -x "$SOURCE_LLVM_LD" ]] || { error "DPC++ requires source-built LLVM lld: $SOURCE_LLVM_LD"; exit 1; }
  [[ -f "$SOURCE_LLVM_DIR/LLVMConfig.cmake" ]] || { error "DPC++ requires source-built LLVMConfig.cmake: $SOURCE_LLVM_DIR/LLVMConfig.cmake"; exit 1; }
  [[ -f "$SOURCE_CLANG_DIR/ClangConfig.cmake" ]] || { error "DPC++ requires source-built ClangConfig.cmake: $SOURCE_CLANG_DIR/ClangConfig.cmake"; exit 1; }
  [[ -f "$IMPL_DIR/tbb/lib/cmake/TBB/TBBConfig.cmake" ]] || { error "DPC++ requires oneTBB at $IMPL_DIR/tbb"; exit 1; }

  export CC="$SOURCE_LLVM_CLANG"
  export CXX="$SOURCE_LLVM_CLANGXX"
  export LD="$SOURCE_LLVM_LD"
  export LLVM_DIR="$SOURCE_LLVM_DIR"
  export CLANG_DIR="$SOURCE_CLANG_DIR"
  export CLANG="$SOURCE_LLVM_CLANG"
  export CLANGXX="$SOURCE_LLVM_CLANGXX"

  info "Using source LLVM for DPC++"
  info "CC=$SOURCE_LLVM_CLANG"
  info "CXX=$SOURCE_LLVM_CLANGXX"
  info "LD=$SOURCE_LLVM_LD"
  info "LLVM_DIR=$SOURCE_LLVM_DIR"
  info "CLANG_DIR=$SOURCE_CLANG_DIR"

  if [[ ! -d "$SOURCES_DIR/dpc++-src" ]]; then
    cd "$SOURCES_DIR"
    git clone https://github.com/intel/llvm dpc++-src
  fi

  cd "$SOURCES_DIR/dpc++-src"
  git fetch origin sycl

  local dpcpp_git_ref="${DPCPP_GIT_REF:-}"
  if [[ -n "$dpcpp_git_ref" ]]; then
    info "Checking out pinned DPC++ ref: $dpcpp_git_ref"
    git checkout -f "$dpcpp_git_ref"
  else
    warning "DPCPP_GIT_REF not set; using branch 'sycl' HEAD"
    git checkout -f sycl
    git pull --ff-only origin sycl
  fi

  local local_jobs="${DPCPP_NCORES:-32}"

  if [[ "${BACKENDS:-}" == *"openmp"* ]]; then
    rm -rf build
    python3 buildbot/configure.py \
      --native_cpu \
      --cmake-opt="-DSYCL_ENABLE_TBB=ON" \
      --cmake-opt="-DTBB_DIR=$IMPL_DIR/tbb/lib/cmake/TBB"

    python3 buildbot/compile.py -j "$local_jobs"
    rm -rf "$IMPL_DIR/dpc++-cpu"
    mv build/install "$IMPL_DIR/dpc++-cpu"
    rm -rf build
  fi

  if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
    local dpcpp_cuda_root="${DPCPP_CUDA_PATH:-${CUDA_PATH:-}}"
    [[ -n "$dpcpp_cuda_root" ]] || {
      error "DPC++ CUDA build requested but neither DPCPP_CUDA_PATH nor CUDA_PATH is set"
      exit 1
    }

    local dpcpp_cupti_include="${DPCPP_CUPTI_INCLUDE_DIR:-}"
    local dpcpp_cupti_library="${DPCPP_CUPTI_LIBRARY:-}"

    if [[ -z "$dpcpp_cupti_include" && -f "$dpcpp_cuda_root/extras/CUPTI/include/cupti.h" ]]; then
      dpcpp_cupti_include="$dpcpp_cuda_root/extras/CUPTI/include"
    fi

    if [[ -z "$dpcpp_cupti_library" && -f "$dpcpp_cuda_root/extras/CUPTI/lib64/libcupti.so" ]]; then
      dpcpp_cupti_library="$dpcpp_cuda_root/extras/CUPTI/lib64/libcupti.so"
    fi

    info "Using DPC++ CUDA toolkit root: $dpcpp_cuda_root"
    info "Using DPC++ CUPTI include: ${dpcpp_cupti_include:-<unset>}"
    info "Using DPC++ CUPTI library: ${dpcpp_cupti_library:-<unset>}"

    rm -rf build
    python3 buildbot/configure.py \
      --native_cpu \
      --cuda \
      --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=$dpcpp_cuda_root" \
      --cmake-opt="-DCUDAToolkit_ROOT=$dpcpp_cuda_root" \
      --cmake-opt="-DSYCL_ENABLE_TBB=ON" \
      --cmake-opt="-DTBB_DIR=$IMPL_DIR/tbb/lib/cmake/TBB" \
      ${dpcpp_cupti_include:+--cmake-opt=-DCUDAToolkit_CUPTI_INCLUDE_DIR=$dpcpp_cupti_include} \
      ${dpcpp_cupti_library:+--cmake-opt=-DCUDAToolkit_CUPTI_LIBRARY=$dpcpp_cupti_library}

    python3 buildbot/compile.py -j "$local_jobs"
    rm -rf "$IMPL_DIR/dpc++-cuda"
    mv build/install "$IMPL_DIR/dpc++-cuda"
    rm -rf build
  fi

  if [[ "${BACKENDS:-}" == *"hip"* ]]; then
    : "${ROCM_PATH:?ROCM_PATH is required for HIP DPC++ builds}"

    rm -rf build
    python3 buildbot/configure.py \
      --native_cpu \
      --hip \
      --cmake-opt="-DSYCL_BUILD_PI_HIP_ROCM_DIR=$ROCM_PATH" \
      --cmake-opt="-DSYCL_ENABLE_TBB=ON" \
      --cmake-opt="-DTBB_DIR=$IMPL_DIR/tbb/lib/cmake/TBB" \
      --cmake-opt="-DLIBCLC_TARGETS_TO_BUILD=amdgcn-amd-amdhsa-llvm"
    python3 buildbot/compile.py -j "$local_jobs"
    rm -rf "$IMPL_DIR/dpc++-hip"
    mv build/install "$IMPL_DIR/dpc++-hip"
    rm -rf build
  fi

  if [[ "${BACKENDS:-}" == *"levelzero"* ]]; then
    rm -rf build
    python3 buildbot/configure.py \
      --native_cpu \
      --cmake-opt="-DLEVEL_ZERO_LIBRARY=$ONEAPI_ROOT/2025.2/lib:/usr/lib64" \
      --cmake-opt="-DLEVEL_ZERO_INCLUDE=/usr/include/level_zero" \
      --cmake-opt="-DSYCL_ENABLE_TBB=ON" \
      --cmake-opt="-DTBB_DIR=$IMPL_DIR/tbb/lib/cmake/TBB"
    python3 buildbot/compile.py -j "$local_jobs"
    rm -rf "$IMPL_DIR/dpc++-$BACKENDS"
    mv build/install "$IMPL_DIR/dpc++-$BACKENDS"
    rm -rf build
  fi
}

install_acpp() {
  [[ "$INSTALL_ACPP" == "1" ]] || return 0

  install_source_llvm

  if [[ -d "$IMPL_DIR/adaptivecpp" ]]; then
    info "AdaptiveCpp already installed: $IMPL_DIR/adaptivecpp"
    return
  fi

  [[ -x "$SOURCE_LLVM_CLANG" ]] || { error "AdaptiveCpp requires source-built LLVM clang: $SOURCE_LLVM_CLANG"; exit 1; }
  [[ -x "$SOURCE_LLVM_CLANGXX" ]] || { error "AdaptiveCpp requires source-built LLVM clang++: $SOURCE_LLVM_CLANGXX"; exit 1; }
  [[ -f "$SOURCE_LLVM_DIR/LLVMConfig.cmake" ]] || { error "AdaptiveCpp requires source-built LLVMConfig.cmake"; exit 1; }
  [[ -f "$SOURCE_LLVM_LIB" ]] || {
    error "AdaptiveCpp requires source-built libLLVM.so: $SOURCE_LLVM_LIB"
    exit 1
  }

  experiment "Building and installing AdaptiveCpp"
  cd "$SYCL_ROOT"

  if [[ ! -d "$SOURCES_DIR/adaptivecpp-src" ]]; then
    cd "$SOURCES_DIR"
    git clone https://github.com/AdaptiveCpp/AdaptiveCpp adaptivecpp-src
  fi

  cd "$SOURCES_DIR/adaptivecpp-src"
  rm -rf build
  mkdir -p build
  cd build

  export ACPP_NVPTX_CLANG="$SOURCE_LLVM_CLANGXX"
  export PATH="$SOURCE_LLVM_BIN:$PATH"

  local acpp_extra=()
  if [[ "${BACKENDS:-}" == *"hip"* ]]; then
    acpp_extra+=("-DACPP_EXPERIMENTAL_LLVM=ON")
  fi

  experiment "Using the following for AdaptiveCpp"
  info "SOURCE_LLVM_INSTALL_ROOT=$SOURCE_LLVM_INSTALL_ROOT"
  info "CC=$SOURCE_LLVM_CLANG"
  info "CXX=$SOURCE_LLVM_CLANGXX"
  info "LLVM_DIR=$SOURCE_LLVM_DIR"
  info "libLLVM=$SOURCE_LLVM_LIB"
  info "ROCM_PATH=${ROCM_PATH:-}"
  info "CUDA_PATH=${CUDA_PATH:-}"

  local acpp_cmake_args=(
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_C_COMPILER="$SOURCE_LLVM_CLANG"
    -D CMAKE_CXX_COMPILER="$SOURCE_LLVM_CLANGXX"
    -D LLVM_DIR="$SOURCE_LLVM_DIR"
    -D WITH_SSCP_COMPILER=ON
    -D CMAKE_INSTALL_PREFIX="$IMPL_DIR/adaptivecpp"
  )

  if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
    : "${CUDA_PATH:?CUDA_PATH is required when BACKENDS includes cuda}"
    acpp_cmake_args+=(
      -D CUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"
      -D CUDAToolkit_ROOT="$CUDA_PATH"
    )
  fi
  if [[ "${BACKENDS:-}" == *"hip"* ]]; then
    : "${ROCM_PATH:?ROCM_PATH is required when BACKENDS includes hip}"
  fi
  acpp_cmake_args+=("${acpp_extra[@]}")
  cmake .. "${acpp_cmake_args[@]}"

  make -j "$NCORES" install
  cd ..
  rm -rf build
}

install_unisycl
install_source_llvm
install_dpcpp
install_acpp

experiment "All requested installs are complete."
