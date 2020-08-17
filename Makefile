NVCC := /usr/local/cuda/bin/nvcc
BUILD_DIR := build
SRC := src

all:
	mkdir -p ${BUILD_DIR}
	${NVCC} ${SRC}/arr_add_host.cu -o ${BUILD_DIR}/arr_add_cpu
	${NVCC} ${SRC}/arr_add_gpu.cu -o ${BUILD_DIR}/arr_add_gpu
	${NVCC} ${SRC}/get_gpu_info.cu -o ${BUILD_DIR}/get_gpu

clean:
	rm -rf ${BUILD_DIR}/