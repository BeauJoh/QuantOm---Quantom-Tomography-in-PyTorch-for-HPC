.PHONY: all plot plot-strong plot-weak plot-frs \
        install reinstall \
        env env-pytorch env-check bootstrap bootstrap-all \
        build build-cpp build-omp build-sycl \
        build-sycl-acpp build-sycl-tbb build-sycl-cuda build-sycl-hip build-sycl-xpu \
        rerun rerun-strong rerun-weak rerun-frs \
        rerun-frs-cpu rerun-frs-cuda rerun-frs-hip rerun-frs-xpu \
        clean clean-cpp clean-omp clean-sycl \
        smoke scaling profile debug \
        test test-cpp test-omp test-sycl \
        test-omp-basic test-omp-instrument test-omp-flto test-omp-instrument-flto \
        test-sycl-acpp test-sycl-acpp-omp test-sycl-acpp-cuda \
        test-sycl-dpcpp-cuda test-sycl-dpcpp-hip test-sycl-usy

SHELL := /usr/bin/env bash
PIXI_RUN := ./setup-pixi.sh run

# Default: regenerate figures from results/ only
all: plot

# -------------------------
# Plotting
# -------------------------

plot: plot-strong plot-weak plot-frs

plot-strong:
	./plot_strong_scaling.sh

plot-weak:
	./plot_weak_scaling.sh

plot-frs:
	./plot_fixed_resource_and_stacked.sh

# -------------------------
# Pixi / Python environment
# -------------------------

env:
	./setup-pixi.sh install

env-pytorch: env
	./setup-pytorch.sh

env-check: env-pytorch
	./setup-pixi.sh run python -c "import sys, torch; print(sys.executable); print(torch.__version__)"

bootstrap: env-pytorch build

bootstrap-all: env-pytorch build plot

# -------------------------
# Install toolchains / runtimes
# -------------------------

install:
	cd sycl && ./install.sh

reinstall:
	cd sycl && REINSTALL=1 ./install.sh

# -------------------------
# Build Python wrappers
# -------------------------

# Common CPU-side workflow:
# - cpp wrapper
# - omp wrapper
# - SYCL AdaptiveCpp wrapper
# - SYCL DPC++ TBB wrapper
build: env build-cpp build-omp build-sycl

build-cpp:
	$(PIXI_RUN) make -C cpp CXX=g++ CC=gcc

build-omp:
	$(PIXI_RUN) make -C omp CXX=g++ CC=gcc

build-sycl: build-sycl-acpp build-sycl-tbb

build-sycl-acpp:
	$(PIXI_RUN) make -C sycl CXX=g++ CC=gcc

build-sycl-tbb:
	$(PIXI_RUN) make -C sycl dpc++_for_tbb CXX=g++ CC=gcc

build-sycl-cuda:
	$(PIXI_RUN) make -C sycl dpc++_for_cuda CXX=g++ CC=gcc

build-sycl-hip:
	$(PIXI_RUN) make -C sycl dpc++_for_hip CXX=g++ CC=gcc

build-sycl-xpu:
	$(PIXI_RUN) make -C sycl dpc++_for_spirv CXX=g++ CC=gcc

# -------------------------
# Backend detection helper
# -------------------------

define get_backends
source ./setup-backends.sh >/dev/null 2>&1 && printf '%s' "$$BACKENDS"
endef

# -------------------------
# Rerun experiments into results/
# -------------------------

rerun: rerun-strong rerun-weak rerun-frs

rerun-strong: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 ./plot_strong_scaling.sh

rerun-weak: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 ./plot_weak_scaling.sh

rerun-frs: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

# Optional explicit per-backend entry points
rerun-frs-cpu: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-cuda: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 FORCE_CUDA=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-hip: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 FORCE_HIP=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-xpu: env-pytorch build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 FORCE_XPU=1 ./plot_fixed_resource_and_stacked.sh

# -------------------------
# Tests
# -------------------------

test: env test-omp test-sycl

test-cpp:
	@echo "No explicit cpp test target defined."

test-omp: test-omp-basic

test-omp-basic:
	$(PIXI_RUN) make -C omp omp_test

test-omp-instrument:
	$(PIXI_RUN) make -C omp omp_instrument

test-omp-flto:
	$(PIXI_RUN) make -C omp omp_test_flto

test-omp-instrument-flto:
	$(PIXI_RUN) make -C omp omp_instrument_flto

test-sycl: test-sycl-acpp test-sycl-dpcpp-cuda test-sycl-dpcpp-hip

test-sycl-acpp:
	$(PIXI_RUN) make -C sycl sycl_test_acpp

test-sycl-acpp-omp:
	$(PIXI_RUN) make -C sycl sycl_test_acpp_omp

test-sycl-acpp-cuda:
	$(PIXI_RUN) make -C sycl sycl_test_acpp_cud

test-sycl-dpcpp-cuda:
	$(PIXI_RUN) make -C sycl sycl_test_dpcpp_cuda

test-sycl-dpcpp-hip:
	$(PIXI_RUN) make -C sycl sycl_test_dpcpp_hip

test-sycl-usy:
	$(PIXI_RUN) make -C sycl sycl_test_usy

# -------------------------
# Legacy helper scripts
# -------------------------

smoke:
	./run.sh --smoke

scaling:
	./run_omp_thread_scaling.sh

profile:
	./profile_2dloits_performance.sh

debug:
	./debug.sh

# -------------------------
# Cleanup
# -------------------------

clean: clean-cpp clean-omp clean-sycl

clean-cpp:
	$(PIXI_RUN) make -C cpp clean

clean-omp:
	$(PIXI_RUN) make -C omp clean

clean-sycl:
	$(PIXI_RUN) make -C sycl clean
