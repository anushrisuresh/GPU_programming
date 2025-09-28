//cuda programming, memory assignment

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

// ---------- Device kernels (<=40 lines each) ----------

// Global-only: y[i] = a*x[i] + b (a,b passed as args). No shared/const.
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

// Constant-only: y[i] = c_params[0]*x[i] + c_params[1]
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

// Shared+constant: tile loads to shared mem, then compute + block reduce.
// scratch[blockIdx.x] will store sum(y) per block (for demo).
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
    float xi = x[i];       // register
    float yi = a * xi + b; // register
    y[i] = yi;             // global store
    val = yi;
  }
  tile[tid] = val;  // shared store
  __syncthreads();

  // In-place block reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] += tile[tid + s];
    __syncthreads();
  }
  if (tid == 0) scratch[blockIdx.x] = tile[0]; // global write
}

// Utility: simple grid-stride to read per-block sums (optional)
__global__ void finalize_reduce(const float* __restrict__ scratch,
                                float* __restrict__ out, int nblocks) {
  float sum = 0.f;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) sum += scratch[i];
  if (threadIdx.x == 0) *out = sum;
}

// ---------- Host utilities (<=40 lines each) ----------

struct Args {
  int64_t N = 1 << 20;
  int block = 256;
  std::string variant = "all";
  int repeats = 3;
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

static void init_host(std::vector<float>& x,
                      std::vector<float>& y,
                      float& a, float& b) {
  a = 2.25f; b = -1.5f;
  for (size_t i = 0; i < x.size(); ++i) x[i] = float(i % 1000) * 0.5f;
  std::fill(y.begin(), y.end(), 0.f);
}

static void verify(const std::vector<float>& x,
                   const std::vector<float>& y,
                   float a, float b) {
  // Basic correctness check on a sample of points
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

// Thin wrappers to unify kernel signatures for timing
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

// ---------- Main (<=40 lines) ----------

int main(int argc, char** argv) {
  Args args; parse_args(argc, argv, args);

  const int64_t N = args.N;
  const int B = args.block;
  const int G = int((N + B - 1) / B);
  const size_t bytes = N * sizeof(float);

  std::vector<float> xh(N), yh(N);
  float a, b; init_host(xh, yh, a, b);

  float *xd=nullptr, *yd=nullptr, *scratch=nullptr, *sum=nullptr;
  CUDA_OK(cudaMalloc(&xd, bytes));
  CUDA_OK(cudaMalloc(&yd, bytes));
  CUDA_OK(cudaMalloc(&scratch, G * sizeof(float)));
  CUDA_OK(cudaMalloc(&sum, sizeof(float)));
  CUDA_OK(cudaMemcpy(xd, xh.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(yd, yh.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpyToSymbol(c_params, &a, sizeof(float), 0));
  CUDA_OK(cudaMemcpyToSymbol(c_params, &b, sizeof(float), sizeof(float)));

  printf("N=%lld, block=%d, grid=%d, variant=%s, reps=%d\n",
         (long long)N, B, G, args.variant.c_str(), args.repeats);

  auto run_one = [&](const char* name,
                     float (*timed)(void (*)(int,int,int,int,
                                             const float*,float*,float*,
                                             int,float,float),
                                    int,int,int,int,
                                    const float*,float*,float*,int,float,float),
                     void (*launcher)(int,int,int,int,
                                      const float*,float*,float*,int,float,float),
                     int shmem)->void {
    CUDA_OK(cudaMemset(yd, 0, bytes));
    float ms = timed(launcher, G, B, shmem, args.repeats,
                     xd, yd, scratch, (int)N, a, b);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    if (strcmp(name,"shared")==0) {
      finalize_reduce<<<1, 256>>>(scratch, sum, G);
      CUDA_OK(cudaDeviceSynchronize());
    }
    CUDA_OK(cudaMemcpy(yh.data(), yd, bytes, cudaMemcpyDeviceToHost));
    verify(xh, yh, a, b);
    printf("%-8s avg %.3f ms\n", name, ms);
  };

  if (args.variant=="global" || args.variant=="all")
    run_one("global", time_ms, launch_global, 0);
  if (args.variant=="constant" || args.variant=="all")
    run_one("constant", time_ms, launch_constant, 0);
  if (args.variant=="shared" || args.variant=="all") {
    int sh = B * (int)sizeof(float);
    run_one("shared", time_ms, launch_shared, sh);
  }

  CUDA_OK(cudaFree(xd)); CUDA_OK(cudaFree(yd));
  CUDA_OK(cudaFree(scratch)); CUDA_OK(cudaFree(sum));
  printf("Done.\n");
  return 0;
}