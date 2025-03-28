# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -gencode arch=compute_75,code=sm_75

# C++ Compiler
CXX = g++
CXX_FLAGS = -std=c++11

# Target executable
TARGET = main

# Source files
CUDA_SRC = main.cu
CPP_SRC = test_bmp.cpp

# Object files
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
CPP_OBJ = $(CPP_SRC:.cpp=.o)

# Build rule
all: $(TARGET)

$(TARGET): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu test_bmp.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp test_bmp.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

clean:
	rm -f $(CUDA_OBJ) $(CPP_OBJ) $(TARGET)

.PHONY: all clean