#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# shellcheck disable=SC1091
source "../setup-backends.sh"
# shellcheck disable=SC1091
source "../utils.sh"

TOP_LEVEL="${TOP_LEVEL:-${PWD:-$(pwd)}}"
HOST="${HOST:-$(hostname -s)}"
IMPL_DIR="${IMPL_DIR:-$TOP_LEVEL/sycl-implementations/$HOST}"
SOURCES_DIR="${SOURCES_DIR:-$TOP_LEVEL/sycl-implementations/sources}"
SOURCE_LLVM_VERSION="${SOURCE_LLVM_VERSION:-19.0.1}"
SOURCE_LLVM_MAJOR="${SOURCE_LLVM_MAJOR:-19}"
INSTALL_DIR="${INSTALL_DIR:-$IMPL_DIR/llvm-${SOURCE_LLVM_VERSION}}"

export TOP_LEVEL HOST IMPL_DIR SOURCES_DIR INSTALL_DIR SOURCE_LLVM_VERSION SOURCE_LLVM_MAJOR

mkdir -p "$SOURCES_DIR"

experiment "Preparing source LLVM build"
info "INSTALL_DIR=$INSTALL_DIR"
info "SOURCES_DIR=$SOURCES_DIR"
info "BACKENDS=${BACKENDS:-}"
info "SOURCE_LLVM_VERSION=$SOURCE_LLVM_VERSION"

cd "$SOURCES_DIR"
rm -rf build

LLVM_SRC_DIR="llvm-${SOURCE_LLVM_VERSION}"
LLVM_TARBALL_XZ="llvm-project-${SOURCE_LLVM_VERSION}.src.tar.xz"
LLVM_TARBALL_GZ="${LLVM_SRC_DIR}.tar.gz"

download_with_fallbacks() {
  local version="$1"

  # 1) Try the old releases site layout
  local url1="https://releases.llvm.org/${version}/llvm-project-${version}.src.tar.xz"

  # 2) Try GitHub codeload tag archive
  local url2="https://codeload.github.com/llvm/llvm-project/tar.gz/refs/tags/llvmorg-${version}"

  # 3) Last resort: shallow clone release branch
  local branch="release/${SOURCE_LLVM_MAJOR}.x"

  if wget -O "$LLVM_TARBALL_XZ" "$url1"; then
    tar -xvf "$LLVM_TARBALL_XZ"
    rm -f "$LLVM_TARBALL_XZ"
    mv "llvm-project-${version}.src" "$LLVM_SRC_DIR"
    return 0
  fi

  warning "Primary release tarball failed for LLVM ${version}"

  if wget -O "$LLVM_TARBALL_GZ" "$url2"; then
    tar -xvf "$LLVM_TARBALL_GZ"
    rm -f "$LLVM_TARBALL_GZ"

    # codeload expands to llvm-project-llvmorg-<version>
    local extracted_dir="llvm-project-llvmorg-${version}"
    if [[ -d "$extracted_dir" ]]; then
      mv "$extracted_dir" "$LLVM_SRC_DIR"
      return 0
    fi

    error "Expected extracted directory '$extracted_dir' was not found"
    return 1
  fi

  warning "GitHub tag archive also failed for LLVM ${version}"
  warning "Falling back to shallow clone of ${branch}"

  git clone --depth 1 --branch "$branch" https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
}

if [[ ! -d "$LLVM_SRC_DIR" ]]; then
  experiment "Fetching LLVM ${SOURCE_LLVM_VERSION}"
  download_with_fallbacks "$SOURCE_LLVM_VERSION"
fi

LLVM_TARGETS="X86"
LLVM_EXTRA_ARGS=()

if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
  LLVM_TARGETS="${LLVM_TARGETS};NVPTX"
  if [[ -n "${CUDA_PATH:-}" ]]; then
    LLVM_EXTRA_ARGS+=("-DCUDAToolkit_ROOT=${CUDA_PATH}")
    LLVM_EXTRA_ARGS+=("-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}")
  fi
  if [[ -n "${CUDA_DEV_TARGET:-}" ]]; then
    LLVM_EXTRA_ARGS+=("-DLIBOMPTARGET_DEVICE_ARCHITECTURES=${CUDA_DEV_TARGET}")
  fi
fi

if [[ "${BACKENDS:-}" == *"hip"* ]]; then
  LLVM_TARGETS="${LLVM_TARGETS};AMDGPU"
  if [[ -n "${ROCM_PATH:-}" ]]; then
    LLVM_EXTRA_ARGS+=("-DROCM_PATH=${ROCM_PATH}")
  fi
  if [[ -n "${HIP_DEV_TARGET:-}" ]]; then
    LLVM_EXTRA_ARGS+=("-DLIBOMPTARGET_DEVICE_ARCHITECTURES=${HIP_DEV_TARGET}")
  fi
fi

: "${GCC_DIR:?GCC_DIR is not set; check setup-backends.sh}"
[[ -x "$GCC_DIR/bin/gcc" ]] || { error "gcc not found at $GCC_DIR/bin/gcc"; exit 1; }
[[ -x "$GCC_DIR/bin/g++" ]] || { error "g++ not found at $GCC_DIR/bin/g++"; exit 1; }

experiment "Configuring source LLVM"
info "LLVM targets=$LLVM_TARGETS"
info "GCC toolchain=$GCC_DIR"
info "CUDA_PATH=${CUDA_PATH:-}"
info "ROCM_PATH=${ROCM_PATH:-}"

cmake -S "${LLVM_SRC_DIR}/llvm" \
      -B build \
      -G Ninja \
      -D LLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld" \
      -D LLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp;offload;libunwind;compiler-rt" \
      -D LLVM_BUILD_LLVM_DYLIB=ON \
      -D BUILD_SHARED_LIBS=ON \
      -D LLVM_ENABLE_RTTI=ON \
      -D LLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_C_COMPILER="$GCC_DIR/bin/gcc" \
      -D CMAKE_CXX_COMPILER="$GCC_DIR/bin/g++" \
      -D LIBOMPTARGET_DEBUG=1 \
      -D OPENMP_TEST_FLAGS=--gcc-toolchain="$GCC_DIR" \
      -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      "${LLVM_EXTRA_ARGS[@]}"

cd build
time nice ninja -j "${NCORES:-64}" install
cd ..

[[ -x "$INSTALL_DIR/bin/clang" ]] || { error "Installed clang missing at $INSTALL_DIR/bin/clang"; exit 1; }
[[ -x "$INSTALL_DIR/bin/clang++" ]] || { error "Installed clang++ missing at $INSTALL_DIR/bin/clang++"; exit 1; }
[[ -f "$INSTALL_DIR/lib/cmake/llvm/LLVMConfig.cmake" ]] || { error "Installed LLVMConfig.cmake missing"; exit 1; }
[[ -f "$INSTALL_DIR/lib/cmake/clang/ClangConfig.cmake" ]] || { error "Installed ClangConfig.cmake missing"; exit 1; }
[[ -f "$INSTALL_DIR/lib/libLLVM.so" ]] || { error "Installed libLLVM.so missing"; exit 1; }

experiment "Source LLVM install complete"
info "clang=$INSTALL_DIR/bin/clang"
info "clang++=$INSTALL_DIR/bin/clang++"
info "LLVMConfig=$INSTALL_DIR/lib/cmake/llvm/LLVMConfig.cmake"
info "ClangConfig=$INSTALL_DIR/lib/cmake/clang/ClangConfig.cmake"
info "libLLVM=$INSTALL_DIR/lib/libLLVM.so"

rm -rf build

