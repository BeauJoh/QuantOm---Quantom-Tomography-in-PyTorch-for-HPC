.PHONY: all plot plot-strong plot-weak plot-frs \
        install reinstall \
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
build: build-cpp build-omp build-sycl

build-cpp:
	$(MAKE) -C cpp CXX=g++ CC=gcc

build-omp:
	$(MAKE) -C omp CXX=g++ CC=gcc

build-sycl: build-sycl-acpp build-sycl-tbb

build-sycl-acpp:
	$(MAKE) -C sycl CXX=g++ CC=gcc

build-sycl-tbb:
	$(MAKE) -C sycl dpc++_for_tbb CXX=g++ CC=gcc

build-sycl-cuda:
	$(MAKE) -C sycl dpc++_for_cuda CXX=g++ CC=gcc

build-sycl-hip:
	$(MAKE) -C sycl dpc++_for_hip CXX=g++ CC=gcc

build-sycl-xpu:
	$(MAKE) -C sycl dpc++_for_spirv CXX=g++ CC=gcc
# -------------------------
# Backend detection helper
# -------------------------

# setup_backends.sh lives at repo root and exports BACKENDS for the current host.
define get_backends
source ./setup-backends.sh >/dev/null 2>&1 && printf '%s' "$$BACKENDS"
endef

# -------------------------
# Rerun experiments into results/
# -------------------------

rerun: rerun-strong rerun-weak rerun-frs

rerun-strong: build-cpp build-omp build-sycl-acpp build-sycl-tbb
	$(MAKE) -C sycl activate_dpcpp_tbb CXX=g++ CC=gcc
	DORUN=1 ./plot_strong_scaling.sh

rerun-weak: build-cpp build-omp build-sycl-acpp build-sycl-tbb
	$(MAKE) -C sycl activate_dpcpp_tbb CXX=g++ CC=gcc
	DORUN=1 ./plot_weak_scaling.sh

rerun-frs: build-cpp build-omp build-sycl-acpp build-sycl-tbb
	$(MAKE) -C sycl activate_dpcpp_tbb CXX=g++ CC=gcc
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

# Optional explicit per-backend entry points
rerun-frs-cpu: build-cpp build-omp build-sycl-acpp build-sycl-tbb
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-cuda: build-cpp build-omp build-sycl-acpp build-sycl-tbb build-sycl-cuda
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-hip: build-cpp build-omp build-sycl-acpp build-sycl-tbb build-sycl-hip
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

rerun-frs-xpu: build-cpp build-omp build-sycl-acpp build-sycl-tbb build-sycl-xpu
	DORUN=1 ./plot_fixed_resource_and_stacked.sh

# -------------------------
# Tests
# -------------------------

test: test-omp test-sycl

test-cpp:
	@echo "No explicit cpp test target defined."

test-omp: test-omp-basic

test-omp-basic:
	$(MAKE) -C omp omp_test

test-omp-instrument:
	$(MAKE) -C omp omp_instrument

test-omp-flto:
	$(MAKE) -C omp omp_test_flto

test-omp-instrument-flto:
	$(MAKE) -C omp omp_instrument_flto

test-sycl: test-sycl-acpp test-sycl-dpcpp-cuda test-sycl-dpcpp-hip

test-sycl-acpp:
	$(MAKE) -C sycl sycl_test_acpp

test-sycl-acpp-omp:
	$(MAKE) -C sycl sycl_test_acpp_omp

test-sycl-acpp-cuda:
	$(MAKE) -C sycl sycl_test_acpp_cud

test-sycl-dpcpp-cuda:
	$(MAKE) -C sycl sycl_test_dpcpp_cuda

test-sycl-dpcpp-hip:
	$(MAKE) -C sycl sycl_test_dpcpp_hip

test-sycl-usy:
	$(MAKE) -C sycl sycl_test_usy

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
	$(MAKE) -C cpp clean

clean-omp:
	$(MAKE) -C omp clean

clean-sycl:
	$(MAKE) -C sycl clean
