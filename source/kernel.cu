void kernel(
    float * d_dst, const float * d_src, int width, int height, int stride, 
    float sigma_spatial, float sigma_color, int radius, cudaStream_t stream);

__global__ 
__launch_bounds__(128) 
static void bilateral(
    float * __restrict__ dst, const float * __restrict__ src, 
    int width, int height, int stride, 
    float sigma_spatial, float sigma_color, int radius) {

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    float center = src[y * stride + x];

    float num { 0.f };
    float den { 0.f };

    for (int cy = max(y - radius, 0); cy <= min(y + radius, height - 1); ++cy) {
        for (int cx = max(x - radius, 0); cx <= min(x + radius, width - 1); ++cx) {
            const float value = src[cy * stride + cx];

            float space = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            float range = (value - center) * (value - center);

            float weight = expf(space * sigma_spatial + range * sigma_color);

            num += weight * value;
            den += weight;
        }
    }

    dst[y * stride + x] = num / den;
}

void kernel(
    float * d_dst, const float * d_src, int width, int height, int stride,
    float sigma_spatial, float sigma_color, int radius, cudaStream_t stream) {

    dim3 blocks = dim3(16, 8);
    dim3 grids = dim3((width - 1) / blocks.x + 1, (height - 1) / blocks.y + 1);

    bilateral<<<grids, blocks, 0, stream>>>(
        d_dst, d_src, width, height, stride, sigma_spatial, sigma_color, radius);
}
