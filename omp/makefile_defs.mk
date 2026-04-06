IMPL_DIR ?= `pwd`/sycl-implementations/$(HOST)
IRIS_INSTALL_ROOT ?= $(IMPL_DIR)/iris
IRIS=$(IRIS_INSTALL_ROOT)

LIBTORCH ?= `python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`/../..
TORCH_LDFLAGS = -Wl,-rpath,$(LIBTORCH)/lib -L$(LIBTORCH)/lib
TORCH_CXXFLAGS = -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_LIBS = -ltorch -ltorch_cpu -lc10 -ltorch
TORCH_CPPFLAGS = -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include

CC ?= gcc
CXX ?= g++
FORTRAN ?= gfortran
NVCC ?= $(CUDA_PATH)/bin/nvcc
HIPCC ?= $(ROCM_PATH)/bin/hipcc
CHARMSYCL_LDFLAGS ?= -L$(CHARMSYCL)/lib -L$(CHARMSYCL)/lib64 -lcharm -lpthread -ldl
DPCPP_LDFLAGS ?= -Wl,-rpath=$(DPCPP_INSTALL_ROOT)/lib -L$(DPCPP_INSTALL_ROOT)/lib -lsycl

CFLAGS=-I$(IRIS)/include/ #-O3 -std=c99 -DRELEASE_BUILD=1
CXXFLAGS += -I$(IRIS)/include/
FFLAGS=-g -I$(IRIS)/include/iris
LDFLAGS=-L$(IRIS)/lib64 -L$(IRIS)/lib -liris -lpthread -ldl

