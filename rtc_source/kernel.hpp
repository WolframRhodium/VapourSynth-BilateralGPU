const auto kernel_source_template = R"""(
/*
external variables:
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    #define BLOCK_X (int),
    #define BLOCK_Y (int),
    bool use_shared_memory
*/

__device__ static const dim3 BlockDim = dim3(BLOCK_X, BLOCK_Y);

extern "C"
__global__
__launch_bounds__(BLOCK_X * BLOCK_Y)
void bilateral(
    float * __restrict__ dst, const float * __restrict__ src
) {

    const int x = threadIdx.x + blockIdx.x * BlockDim.x;
    const int y = threadIdx.y + blockIdx.y * BlockDim.y;

    float num {};
    float den {};

    if constexpr (use_shared_memory) {
        extern __shared__ float buffer[
            /* (2 * radius + blockDim.y) * (2 * radius + blockDim.x) */];

        for (int cy = threadIdx.y; cy < 2 * radius + BlockDim.y; cy += BlockDim.y) {
            int sy = min(max(cy - static_cast<int>(threadIdx.y) - radius + y, 0), height - 1);
            for (int cx = threadIdx.x; cx < 2 * radius + BlockDim.x; cx += BlockDim.x) {
                int sx = min(max(cx - static_cast<int>(threadIdx.x) - radius + x, 0), width - 1);
                buffer[cy * (2 * radius + BlockDim.x) + cx] = src[sy * stride + sx];
            }
        }

        __syncthreads();

        if (x >= width || y >= height)
            return;

        const float center = src[y * stride + x];

        for (int cy = -radius; cy <= radius; ++cy) {
            int sy = cy + radius + threadIdx.y;

            for (int cx = -radius; cx <= radius; ++cx) {
                int sx = cx + radius + threadIdx.x;

                float value = buffer[sy * (2 * radius + BlockDim.x) + sx];

                float space = cy * cy + cx * cx;
                float range = (value - center) * (value - center);

                float weight = exp2f(space * sigma_spatial_scaled + range * sigma_color_scaled);

                num += weight * value;
                den += weight;
            }
        }

        dst[y * stride + x] = num / den;
    } else {
        if (x >= width || y >= height)
            return;

        const float center = src[y * stride + x];

        for (int cy = max(y - radius, 0); cy <= min(y + radius, height - 1); ++cy) {
            for (int cx = max(x - radius, 0); cx <= min(x + radius, width - 1); ++cx) {
                const float value = src[cy * stride + cx];

                float space = (y - cy) * (y - cy) + (x - cx) * (x - cx);
                float range = (value - center) * (value - center);

                float weight = exp2f(space * sigma_spatial_scaled + range * sigma_color_scaled);

                num += weight * value;
                den += weight;
            }
        }
    }

    dst[y * stride + x] = num / den;
}
)""";
