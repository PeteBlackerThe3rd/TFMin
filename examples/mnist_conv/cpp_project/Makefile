# TFMin MNIST convolutional example project makefile

cpp_flags = -std=c++11 -O3 -pthread -DEIGEN_USE_THREADS
generated_code_path = ../tfmin_generated/
BINDIR = bin

all : $(BINDIR) $(BINDIR)/native_test

$(BINDIR):
	mkdir -p $(BINDIR)

# build binary
$(BINDIR)/native_test : test_mnist.cpp $(generated_code_path)mnist_model.cpp
	$(info Building implementation of mnist model)
	@(g++ $(cpp_flags) -o $(BINDIR)/native_test test_mnist.cpp $(generated_code_path)mnist_model.cpp -I $(generated_code_path)) && echo "Build succeeded."

clean :
	$(info cleaning build files)
	@rm -f bin/native_test

