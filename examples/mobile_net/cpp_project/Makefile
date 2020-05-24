# TFMin make file automatically generate by the TFMin library

# -pthread -DEIGEN_USE_THREADS

cpp_flags = -std=c++11 -O3
generated_code_path = ../tfmin_generated/
BINDIR = bin

all : $(BINDIR) $(BINDIR)/native_mobile_net

$(BINDIR):
	mkdir -p $(BINDIR)

# link native binary
$(BINDIR)/native_mobile_net : mobile_net_test.cpp $(generated_code_path)mobile_net.cpp
	$(info Building implementation of MobileNet model)
	@(g++ $(cpp_flags) -o $(BINDIR)/native_mobile_net mobile_net_test.cpp $(generated_code_path)mobile_net.cpp -I $(generated_code_path)) && echo "Build succeeded."

clean :
	$(info cleaning build files)
	@rm -f $(BINDIR)/native_mobile_net

