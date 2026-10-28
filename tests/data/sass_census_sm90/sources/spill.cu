// Fixture source for tests/data/sass_census_sm90/: a kernel that uses local memory, so that its
// real nvdisasm -c listing carries STL/LDL and its cuobjdump -res-usage line a nonzero stack.
__global__ void __launch_bounds__(128) spill_kernel(const double* in, double* out, int n, int k) {
    double acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = in[threadIdx.x + i * 128];
    for (int j = 0; j < n; ++j) acc[(j * 7 + k) & 63] += acc[(j * 3) & 63];
    double r = 0;
    if (threadIdx.x & 1) r = acc[k & 63];
#pragma unroll
    for (int i = 0; i < 64; ++i) r += acc[i];
    out[threadIdx.x] = r;
}
__global__ void second_kernel(int* p) { p[threadIdx.x] = threadIdx.x; }
