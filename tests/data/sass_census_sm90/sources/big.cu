// Fixture source for tests/data/sass_census_sm90/: a kernel whose code passes 64 KiB, so that
// its real nvdisasm -c listing carries five-digit instruction addresses.
__global__ void big_kernel(float* p) {
    float a = p[threadIdx.x];
#pragma unroll
    for (int i = 0; i < 6000; ++i) a = a * 1.0001f + (float)i;
    p[threadIdx.x] = a;
}
