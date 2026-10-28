// A kernel whose listing holds USETMAXREG regions like the decode kernel's (one warpgroup
// releasing to 32 registers, two allocating 104) with more live values in each region than
// the region's registers hold, so ptxas spills in each of them. Built only to produce real
// nvdisasm output of the spill forms (STL, LDL, LDL.LU) on every side of the USETMAXREG
// lines for the census's CPU tests; it is never launched.
//   nvcc -arch=sm_90a -cubin -o regions.cubin regions.cu
template <int N, int S>
__device__ __forceinline__ float churn(const float* __restrict__ in, int t, int n) {
  float v[N];
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = in[t + 384 * i];
  for (int k = 0; k < n; ++k) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = fmaf(v[i], v[(i + S) % N], 1.0f);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < N; ++i) s = fmaf(v[i], (float)(i + 1), s);
  return s;
}

extern "C" __global__ void __launch_bounds__(384, 2)
regions_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int t = threadIdx.x, wg = t / 128;
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
    out[t] = churn<40, 7>(in, t, n);
  } else if (wg == 1) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\n" ::: "memory");
    out[t] = churn<120, 11>(in, t, n);
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\n" ::: "memory");
    out[t] = churn<116, 13>(in, t, n);
  }
}
