constexpr auto kernel_code_raw = R"""(
// #define BLOCK_X 16
// #define BLOCK_Y 8
// #define WIDTH 1920
// #define HEIGHT 1080
// #define STRIDE 1920
// #define SIGMA_SPATIAL_SCALED -0.5f / (sigma_spatial * sigma_spatial) * log2(e)
// #define SIGMA_COLOR_SCALED -0.5f / (sigma_color * sigma_color) * log2(e)
// #define RADIUS 9
// #define USE_SHARED_MEMORY 1

__kernel
__attribute__((reqd_work_group_size(BLOCK_X, BLOCK_Y, 1)))
void bilateral(
    __global float * restrict dst,
    __global const float * restrict src
) {
    const int x = get_local_id(0) + get_group_id(0) * BLOCK_X;
    const int y = get_local_id(1) + get_group_id(1) * BLOCK_Y;

    float num = 0.0f;
    float den = 0.0f;

#if USE_SHARED_MEMORY
    __local float buffer[(2 * RADIUS + BLOCK_Y) * (2 * RADIUS + BLOCK_X)];

    for (int cy = get_local_id(1); cy < 2 * RADIUS + BLOCK_Y; cy += BLOCK_Y) {
        int sy = min(max(cy - (int) get_local_id(1) - RADIUS + y, 0), HEIGHT - 1);
        for (int cx = get_local_id(0); cx < 2 * RADIUS + BLOCK_X; cx += BLOCK_X) {
            int sx = min(max(cx - (int) get_local_id(0) - RADIUS + x, 0), WIDTH - 1);
            buffer[cy * (2 * RADIUS + BLOCK_X) + cx] = src[sy * STRIDE + sx];
        }
    }

    if (x >= WIDTH || y >= HEIGHT)
        return;

    barrier(CLK_LOCAL_MEM_FENCE);

    const float center = src[y * STRIDE + x];

    for (int cy = -RADIUS; cy <= RADIUS; ++cy) {
        int sy = cy + RADIUS + (int) get_local_id(1);

        for (int cx = -RADIUS; cx <= RADIUS; ++cx) {
            int sx = cx + RADIUS + (int) get_local_id(0);

            float value = buffer[sy * (2 * RADIUS + BLOCK_X) + sx];

            float space = cy * cy + cx * cx;
            float range = (value - center) * (value - center);

            float weight = native_exp2(space * SIGMA_SPATIAL_SCALED + range * SIGMA_COLOR_SCALED);

            num += weight * value;
            den += weight;
        }
    }
#else // USE_SHARED_MEMORY
    if (x >= WIDTH || y >= HEIGHT)
        return;

    const float center = src[y * STRIDE + x];

    for (int cy = max(y - RADIUS, 0); cy <= min(y + RADIUS, HEIGHT - 1); ++cy) {
        for (int cx = max(x - RADIUS, 0); cx <= min(x + RADIUS, WIDTH - 1); ++cx) {
            const float value = src[cy * STRIDE + cx];

            float space = (y - cy) * (y - cy) + (x - cx) * (x - cx);
            float range = (value - center) * (value - center);

            float weight = native_exp2(space * SIGMA_SPATIAL_SCALED + range * SIGMA_COLOR_SCALED);

            num += weight * value;
            den += weight;
        }
    }
#endif // USE_SHARED_MEMORY

    dst[y * STRIDE + x] = num / den;
}
)""";
