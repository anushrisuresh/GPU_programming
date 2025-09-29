// CUDA Memory Programming Demo
// Demonstrates different memory types: global, constant, shared, and registers
// Implements SAXPY (y = a*x + b) using various memory access patterns

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>

#define CUDA_OK(call) do {                                           \
  cudaError_t e = (call);                                            \
  if (e != cudaSuccess) {                                            \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
            __FILE__, __LINE__, cudaGetErrorString(e));              \
    exit(1);                                                         \
  }                                                                  \
} while(0)

// ---------- Constant memory (device) ----------
__constant__ float c_params[2]; // c_params[0]=a, c_params[1]=b

// ---------- Device kernels ----------

// Kernel 1: Global memory only - parameters passed as function arguments
// Demonstrates basic global memory access patterns
__global__ void saxpy_global(const float* __restrict__ x,
                             float* __restrict__ y,
                             int n, float a, float b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // register use: locals typically map to registers
    float xi = x[i];
    float yi = a * xi + b;
    y[i] = yi;
  }
}

// Kernel 2: Constant memory - parameters stored in device constant memory
// Constant memory is cached and read-only, good for broadcast values
__global__ void saxpy_constant(const float* __restrict__ x,
                               float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = c_params[0]; // register
    float b = c_params[1]; // register
    float xi = x[i];
    y[i] = a * xi + b;
  }
}

// Kernel 3: Shared memory + constant memory + block reduction
// Demonstrates shared memory usage for inter-thread communication within blocks
// Also performs block-level reduction to compute sum per block
__global__ void saxpy_shared(const float* __restrict__ x,
                             float* __restrict__ y,
                             float* __restrict__ scratch,
                             int n) {
  extern __shared__ float tile[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float a = c_params[0], b = c_params[1]; // registers

  float val = 0.f;
  if (i < n) {
    float xi = x[i];       // load from global memory to register
    float yi = a * xi + b; // computation in registers
    y[i] = yi;             // store result to global memory
    val = yi;
  }
  tile[tid] = val;  // store to shared memory for reduction
  __syncthreads();  // ensure all threads have written to shared memory

  // Tree-based block reduction in shared memory (logarithmic complexity)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] += tile[tid + s];
    __syncthreads();
  }
  if (tid == 0) scratch[blockIdx.x] = tile[0]; // global write
}

// Utility kernel: reduces per-block sums to a single final sum
// Uses grid-stride loop pattern for efficient reduction across blocks
__global__ void finalize_reduce(const float* __restrict__ scratch,
                                float* __restrict__ out, int nblocks) {
  float sum = 0.f;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) sum += scratch[i];
  if (threadIdx.x == 0) *out = sum;
}

// ---------- Host utilities ----------

// Command-line argument structure with defaults
struct Args {
  int64_t N = 1 << 20;        // Default array size: 1M elements
  int block = 256;            // Default threads per block
  std::string variant = "all"; // Which kernel variants to run
  int repeats = 3;            // Number of timing repetitions
};

static void parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc) a.N = atoll(argv[++i]);
    else if (!strcmp(argv[i], "-b") && i + 1 < argc) a.block = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-k") && i + 1 < argc) a.variant = argv[++i];
    else if (!strcmp(argv[i], "-r") && i + 1 < argc) a.repeats = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      printf("Usage: %s [-n N] [-b BLOCK] [-k VAR] [-r REPEATS]\n", argv[0]);
      exit(0);
    }
  }
}

// Initialize host arrays with test data
static void init_host(std::vector<float>& x,
                      std::vector<float>& y,
                      float& a, float& b) {
  a = 2.25f; b = -1.5f;  // SAXPY parameters
  for (size_t i = 0; i < x.size(); ++i) x[i] = float(i % 1000) * 0.5f;  // periodic pattern
  std::fill(y.begin(), y.end(), 0.f);  // zero output array
}

// Verify kernel correctness by checking a sample of computed values
static void verify(const std::vector<float>& x,
                   const std::vector<float>& y,
                   float a, float b) {
  // Sample check (not exhaustive) for performance
  for (size_t i = 0; i < x.size(); i += x.size() / 7 + 1) {
    float expect = a * x[i] + b;
    float diff = fabsf(expect - y[i]);
    if (diff > 1e-3f) {
      fprintf(stderr, "Mismatch at %zu: got %.4f expect %.4f\n",
              i, y[i], expect);
      exit(2);
    }
  }
}

// Timing utility using CUDA events for accurate GPU timing
static float time_ms(void (*launch)(int,int,int,int,
                                    const float*,float*,
                                    float*,int,float,float),
                     int grid, int block, int shmem, int reps,
                     const float* xd, float* yd, float* sd,
                     int n, float a, float b) {
  cudaEvent_t beg, end;
  CUDA_OK(cudaEventCreate(&beg));
  CUDA_OK(cudaEventCreate(&end));
  CUDA_OK(cudaEventRecord(beg));
  for (int r = 0; r < reps; ++r)
    launch(grid, block, shmem, 0, xd, yd, sd, n, a, b);
  CUDA_OK(cudaEventRecord(end));
  CUDA_OK(cudaEventSynchronize(end));
  float ms = 0.f;
  CUDA_OK(cudaEventElapsedTime(&ms, beg, end));
  CUDA_OK(cudaEventDestroy(beg));
  CUDA_OK(cudaEventDestroy(end));
  return ms / reps;
}

// Kernel launcher wrappers - unify different kernel signatures for timing
static void launch_global(int g,int b,int sm, int,
                          const float* x, float* y, float*, int n,
                          float a, float b2) {
  (void)sm;
  saxpy_global<<<g, b>>>(x, y, n, a, b2);
}

static void launch_constant(int g,int b,int sm, int,
                            const float* x, float* y, float*, int n,
                            float, float) {
  (void)sm;
  saxpy_constant<<<g, b>>>(x, y, n);
}

static void launch_shared(int g,int b,int sm, int,
                          const float* x, float* y, float* s, int n,
                          float, float) {
  saxpy_shared<<<g, b, sm>>>(x, y, s, n);
}

// Register usage demonstration kernel
// Shows how local variables map to fast register storage
__global__ void register_demo_kernel() {
    // Multiple local variables - typically stored in registers
    float a=1.1f, b=2.2f, c=3.3f, d=4.4f, e=5.5f;
    float f = a*b + c*d + e;  // computation stays in registers
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("Register demo: computed f=%.2f (using registers)\n", f);
}

// ---------- Main ----------

int main(int argc, char** argv) {
  Args args; parse_args(argc, argv, args);

  const int64_t N = args.N;
  const int B = args.block;
  const int G = int((N + B - 1) / B);
  const size_t bytes = N * sizeof(float);

  std::vector<float> xh(N), yh(N);
  float a, b; init_host(xh, yh, a, b);

  // Display host memory allocation info
  printf("Host memory allocated: %.2f MB for xh + yh (on CPU)\n",
    (double)(xh.size() * sizeof(float) * 2) / (1024.0 * 1024.0));
  printf("Host sample xh[0]=%.2f, yh[0]=%.2f (before copy)\n",
    xh[0], yh[0]);

  // Demonstrate register usage
  printf("Register demo kernel execution:\n");
  register_demo_kernel<<<1, 1>>>();
  CUDA_OK(cudaDeviceSynchronize());

  // Allocate device memory for all variants
  float *xd=nullptr, *yd=nullptr, *scratch=nullptr, *sum=nullptr;
  CUDA_OK(cudaMalloc(&xd, bytes));           // input array
  CUDA_OK(cudaMalloc(&yd, bytes));           // output array  
  CUDA_OK(cudaMalloc(&scratch, G * sizeof(float)));  // per-block sums
  CUDA_OK(cudaMalloc(&sum, sizeof(float)));  // final sum
  
  // Copy data from host to device
  CUDA_OK(cudaMemcpy(xd, xh.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(yd, yh.data(), bytes, cudaMemcpyHostToDevice));
  
  // Copy SAXPY parameters to constant memory
  CUDA_OK(cudaMemcpyToSymbol(c_params, &a, sizeof(float), 0));
  CUDA_OK(cudaMemcpyToSymbol(c_params, &b, sizeof(float), sizeof(float)));

  printf("N=%lld, block=%d, grid=%d, variant=%s, reps=%d\n",
         (long long)N, B, G, args.variant.c_str(), args.repeats);

  // Lambda function to run and time each kernel variant
  auto run_one = [&](const char* name,
                     float (*timed)(void (*)(int,int,int,int,
                                             const float*,float*,float*,
                                             int,float,float),
                                    int,int,int,int,
                                    const float*,float*,float*,int,float,float),
                     void (*launcher)(int,int,int,int,
                                      const float*,float*,float*,int,float,float),
                     int shmem)->void {
    CUDA_OK(cudaMemset(yd, 0, bytes));  // clear output array
    float ms = timed(launcher, G, B, shmem, args.repeats,
                     xd, yd, scratch, (int)N, a, b);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // For shared memory variant, also compute final reduction
    if (strcmp(name,"shared")==0) {
      finalize_reduce<<<1, 256>>>(scratch, sum, G);
      CUDA_OK(cudaDeviceSynchronize());
    }
    
    // Copy results back and verify correctness
    CUDA_OK(cudaMemcpy(yh.data(), yd, bytes, cudaMemcpyDeviceToHost));
    verify(xh, yh, a, b);
    printf("%-8s avg %.3f ms\n", name, ms);
  };

  // Run selected kernel variants based on command line arguments
  if (args.variant=="global" || args.variant=="all")
    run_one("global", time_ms, launch_global, 0);
  if (args.variant=="constant" || args.variant=="all")
    run_one("constant", time_ms, launch_constant, 0);
  if (args.variant=="shared" || args.variant=="all") {
    int sh = B * (int)sizeof(float);  // shared memory size = block_size * sizeof(float)
    run_one("shared", time_ms, launch_shared, sh);
  }

  // Clean up device memory allocations
  CUDA_OK(cudaFree(xd)); CUDA_OK(cudaFree(yd));
  CUDA_OK(cudaFree(scratch)); CUDA_OK(cudaFree(sum));
  printf("Done.\n");
  return 0;
}