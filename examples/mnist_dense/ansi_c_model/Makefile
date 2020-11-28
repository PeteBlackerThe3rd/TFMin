
# Example mnist ansi-c project makefile

compiler = gcc
c_flags = -O3 -LM

mnist_demo : mnist_main.c mnist_model.c mnist_weights.c
	$(compiler) $(c_flags) mnist_main.c mnist_model.c mnist_weights.c -o mnist_demo
