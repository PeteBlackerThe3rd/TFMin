# TFMin MNIST dense example project makefile

cpp_flags = -std=c++11 -O3
generated_code_path = ../tfmin_generated/
# -pthread -DEIGEN_USE_THREADS
BINDIR = bin

all : $(BINDIR) $(BINDIR)/native_test

$(BINDIR):
	mkdir -p $(BINDIR)

# build native binary
$(BINDIR)/native_test : test_simple_rnn.cpp $(generated_code_path)rnn_model.cpp
	$(info Building implementation of mnist model)
	@(g++ $(cpp_flags) -o $(BINDIR)/native_test test_simple_rnn.cpp $(generated_code_path)rnn_model.cpp -I $(generated_code_path)) && echo "Build succeeded."

clean :
	$(info cleaning build files)
	@rm -f $(BINDIR)/native_test

