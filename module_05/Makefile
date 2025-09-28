# Build to ./build. Keep lines short and flags clear.
CXX := nvcc
TARGET := build/memory
SRC := src/memory.cu

# ptxas -v shows register usage (nice for the "register" rubric note)
NVCCFLAGS := -O3 -std=c++17 -Xptxas -v --use_fast_math

$(TARGET): $(SRC)
	mkdir -p build
	$(CXX) $(NVCCFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -rf build