// Fixture source for tests/data/sass_census_sm90/: a call whose target is named like a general
// register, so that its real nvdisasm -c listing names R12 inside a branch target.
extern "C" __device__ __noinline__ float R12(float x) { return x * x + 1.0f; }
extern "C" __global__ void call_kernel(float* p) { p[threadIdx.x] = R12(p[threadIdx.x]); }
