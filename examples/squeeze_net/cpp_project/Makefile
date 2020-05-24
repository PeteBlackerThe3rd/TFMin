# TFMin make file automatically generate by the TFMin library

cpp_flags = -std=c++11 -O3
generated_code_path = ../tfmin_generated/
# -pthread -DEIGEN_USE_THREADS
BINDIR = bin

all : $(BINDIR) $(BINDIR)/native_squeeze_net

$(BINDIR):
	mkdir -p $(BINDIR)

# link native binary
$(BINDIR)/native_squeeze_net : squeeze_net_test.cpp $(generated_code_path)squeeze_net.cpp
	$(info Building implementation of SqueezeNet model)
	@(g++ $(cpp_flags) -o $(BINDIR)/native_squeeze_net squeeze_net_test.cpp $(generated_code_path)squeeze_net.cpp -I $(generated_code_path)) && echo "Build succeeded."

clean :
	$(info cleaning build files)
	@rm -f $(BINDIR)/native_squeeze_net

