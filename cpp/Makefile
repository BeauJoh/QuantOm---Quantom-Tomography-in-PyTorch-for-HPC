include ./makefile_defs.mk

WRAPPER_NAME:=cpp_wrapper$(shell python3-config --extension-suffix)
all: $(WRAPPER_NAME)

###############################################################################
#                                  Sampler                                    #
###############################################################################
libsampler.so: cpp_sampler.cpp ../common/matrix.cpp ../common/utils.cpp
	@echo compiling $@...
	$(CXX) $(CXXFLAGS) -O3 -DWRAPPER_BUILD=1 -std=c++17 -ffast-math -march=native -shared -fPIC -o $@ $^

$(WRAPPER_NAME): libsampler.so pybind_wrapper.cpp
	@echo compiling $@...
	$(CXX) $(CXXFLAGS) -O3 -march=native \
	 	-Wall -Werror -DWRAPPER_BUILD=1 -shared -std=c++17 -fPIC \
		`python3 -m pybind11 --includes` \
		$(TORCH_CXXFLAGS) $(TORCH_CPPFLAGS)\
		-L. -Wl,-rpath,. $^ \
		-ltorch \
		libsampler.so \
		-o $@ \
		$(TORCH_LDFLAGS) $(TORCH_LIBS)

clean:
	rm -f $(WRAPPER_NAME) libsampler.so

