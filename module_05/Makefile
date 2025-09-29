# Build to ./build.
CXX := nvcc
TARGET := build/memory
SRC := src/memory.cu

# ptxas -v shows register usage
NVCCFLAGS := -O3 -std=c++17 -Xptxas -v --use_fast_math

$(TARGET): $(SRC)
	mkdir -p build
	$(CXX) $(NVCCFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -rf build