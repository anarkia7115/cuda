all: stencil stencil_cpu

stencil: ./stencil.cu
	nvcc -pg -O3 $< -o $@

stencil_cpu: ./stencil_cpu.cpp
	g++ -pg -O3 $< -o $@
