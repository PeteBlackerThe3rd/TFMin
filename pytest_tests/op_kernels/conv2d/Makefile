# TFMin make file automatically generate by the TFMin library

cpp_flags = -std=c++11 -O3
native_flags = -pthread -DEIGEN_USE_THREADS

binary = tmp/test_binary
srcs = test.cpp tmp/test_source_conv2d.cpp

all : $(binary)

$(binary) : $(srcs)
	@(g++ $(cpp_flags) $(native_flags) -o $(binary) $(srcs)) && echo "Build succeeded."
clean :
	$(info cleaning build files)
	@rm -f $(binary)
