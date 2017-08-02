all: hello add block_add thread_add block_thread_add
hello: ./hello_world.cu
	nvcc $< -o $@

add: ./add.cu
	nvcc $< -o $@

block_add: ./block_add.cu
	nvcc $< -o $@

thread_add: ./thread_add.cu
	nvcc $< -o $@

block_thread_add: ./block_thread_add.cu
	nvcc $< -o $@
