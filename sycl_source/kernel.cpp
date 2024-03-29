#include <CL/sycl.hpp>

// #define BLOCK_X 16
// #define BLOCK_Y 16

template <bool use_shared_memory, bool has_ref>
inline static void bilateral(
    float * __restrict__ dst, const float * __restrict__ src,
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    sycl::nd_item<2> it, sycl::local_accessor<float, 1> buffer
) {

    int x = static_cast<int>(it.get_global_id(1));
    int y = static_cast<int>(it.get_global_id(0));

    int block_x = static_cast<int>(it.get_local_range(1));
    int block_y = static_cast<int>(it.get_local_range(0));

    int threadIdx_x = static_cast<int>(it.get_local_id(1));
    int threadIdx_y = static_cast<int>(it.get_local_id(0));

    float num {};
    float den {};

    if constexpr (use_shared_memory) {
        for (int cy = threadIdx_y; cy < 2 * radius + block_y; cy += block_y) {
            int sy = sycl::min(sycl::max(cy - threadIdx_y - radius + y, 0), height - 1);
            for (int cx = threadIdx_x; cx < 2 * radius + block_x; cx += block_x) {
                int sx = sycl::min(sycl::max(cx - threadIdx_x - radius + x, 0), width - 1);
                buffer[cy * (2 * radius + block_x) + cx] = src[sy * stride + sx];
            }
        }

        if constexpr (has_ref) {
            for (int cy = threadIdx_y; cy < 2 * radius + block_y; cy += block_y) {
                int sy = sycl::min(sycl::max(cy - threadIdx_y - radius + y, 0), height - 1);
                for (int cx = threadIdx_x; cx < 2 * radius + block_x; cx += block_x) {
                    int sx = sycl::min(sycl::max(cx - threadIdx_x - radius + x, 0), width - 1);
                    buffer[(2 * radius + block_y + cy) * (2 * radius + block_x) + cx] = src[(height + sy) * stride + sx];
                }
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        if (x >= width || y >= height)
            return;

        const float center = buffer[
            (has_ref * (2 * radius + block_y) + radius + threadIdx_y) * (2 * radius + block_x) +
            radius + threadIdx_x
        ]; // src[(has_ref * height + y) * stride + x];

        for (int cy = -radius; cy <= radius; ++cy) {
            int sy = cy + radius + threadIdx_y;

            for (int cx = -radius; cx <= radius; ++cx) {
                int sx = cx + radius + threadIdx_x;

                float value = buffer[(has_ref * (2 * radius + block_y) + sy) * (2 * radius + block_x) + sx];

                float space = cy * cy + cx * cx;
                float range = (value - center) * (value - center);

                float weight = sycl::exp2(space * sigma_spatial_scaled + range * sigma_color_scaled);

                if constexpr (has_ref) {
                    value = buffer[sy * (2 * radius + block_x) + sx];
                }

                num += weight * value;
                den += weight;
            }
        }
    } else {
        if (x >= width || y >= height)
            return;

        const float center = src[(has_ref * height + y) * stride + x];

        for (int cy = sycl::max(y - radius, 0); cy <= sycl::min(y + radius, height - 1); ++cy) {
            for (int cx = sycl::max(x - radius, 0); cx <= sycl::min(x + radius, width - 1); ++cx) {
                float value = src[(has_ref * height + cy) * stride + cx];

                float space = (y - cy) * (y - cy) + (x - cx) * (x - cx);
                float range = (value - center) * (value - center);

                float weight = sycl::exp2(space * sigma_spatial_scaled + range * sigma_color_scaled);

                if constexpr (has_ref) {
                    value = src[cy * stride + cx];
                }

                num += weight * value;
                den += weight;
            }
        }
    }

    dst[y * stride + x] = num / den;
}

sycl::event launch(
    float * d_dst, float * d_src, float * h_buffer,
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    bool use_shared_memory,
    int block_x, int block_y,
    bool has_ref,
    sycl::queue & stream
) {

    auto memcpy_h_to_d = stream.memcpy(d_src, h_buffer, (1 + has_ref) * height * stride * sizeof(float));

    auto kernel = stream.submit([&](sycl::handler & h) {
        h.depends_on(memcpy_h_to_d);

        sycl::range<2> grid_dims {
            static_cast<size_t>(((height - 1) / block_y + 1) * block_y),
            static_cast<size_t>(((width - 1) / block_x + 1) * block_x)
        };
        sycl::range<2> block_dims { (size_t) block_y, (size_t) block_x };

        sycl::local_accessor<float, 1> buffer(
            use_shared_memory ? (1 + has_ref) * (2 * radius + block_x) * (2 * radius + block_y) : 0,
            h
        );

        if (use_shared_memory) {
            if (has_ref) {
                auto bilateral_kernel = [=](sycl::nd_item<2> it)
                    // [[sycl::reqd_work_group_size(1, BLOCK_Y, BLOCK_X)]]
                    #if defined SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT && SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT
                    [[intel::kernel_args_restrict]]
                    #endif
                {
                    bilateral<true, true>(
                        d_dst, d_src,
                        width, height, stride,
                        sigma_spatial_scaled, sigma_color_scaled, radius,
                        it, buffer
                    );
                };

                h.parallel_for(sycl::nd_range { grid_dims, block_dims }, bilateral_kernel);
            } else {
                auto bilateral_kernel = [=](sycl::nd_item<2> it)
                    // [[sycl::reqd_work_group_size(1, BLOCK_Y, BLOCK_X)]]
                    #if defined SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT && SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT
                    [[intel::kernel_args_restrict]]
                    #endif
                {
                    bilateral<true, false>(
                        d_dst, d_src,
                        width, height, stride,
                        sigma_spatial_scaled, sigma_color_scaled, radius,
                        it, buffer
                    );
                };

                h.parallel_for(sycl::nd_range { grid_dims, block_dims }, bilateral_kernel);
            }
        } else {
            if (has_ref) {
                auto bilateral_kernel = [=](sycl::nd_item<2> it)
                    // [[sycl::reqd_work_group_size(1, BLOCK_Y, BLOCK_X)]]
                    #if defined SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT && SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT
                    [[intel::kernel_args_restrict]]
                    #endif
                {
                    bilateral<false, true>(
                        d_dst, d_src,
                        width, height, stride,
                        sigma_spatial_scaled, sigma_color_scaled, radius,
                        it, buffer
                    );
                };

                h.parallel_for(sycl::nd_range { grid_dims, block_dims }, bilateral_kernel);
            } else {
                auto bilateral_kernel = [=](sycl::nd_item<2> it)
                    // [[sycl::reqd_work_group_size(1, BLOCK_Y, BLOCK_X)]]
                    #if defined SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT && SYCL_EXT_INTEL_KERNEL_ARGS_RESTRICT
                    [[intel::kernel_args_restrict]]
                    #endif
                {
                    bilateral<false, false>(
                        d_dst, d_src,
                        width, height, stride,
                        sigma_spatial_scaled, sigma_color_scaled, radius,
                        it, buffer
                    );
                };

                h.parallel_for(sycl::nd_range { grid_dims, block_dims }, bilateral_kernel);
            }
        }
    });

    auto memcpy_d_to_h = stream.submit([&](sycl::handler & h) {
        h.depends_on(kernel);

        h.memcpy(h_buffer, d_dst, height * stride * sizeof(float));
    });

    return memcpy_d_to_h;
}
