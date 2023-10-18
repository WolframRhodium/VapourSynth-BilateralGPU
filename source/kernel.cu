#include <iterator>

#include <cuda_pipeline_primitives.h>

#define BLOCK_X 16
#define BLOCK_Y 8

cudaGraphExec_t get_graphexec(
    float * d_dst, float * d_src, float * h_buffer,
    int width, int height, int stride,
    float sigma_spatial, float sigma_color, int radius,
    bool use_shared_memory);

template <bool use_shared_memory, bool has_ref>
__global__
__launch_bounds__(BLOCK_X * BLOCK_Y)
static void bilateral(
    float * __restrict__ dst, const float * __restrict__ src,
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius) {

    const int x = threadIdx.x + blockIdx.x * BLOCK_X;
    const int y = threadIdx.y + blockIdx.y * BLOCK_Y;

    float num {};
    float den {};

    if constexpr (use_shared_memory) {
        extern __shared__ float buffer[
            /* (1 + has_ref) * (2 * radius + BLOCK_Y) * (2 * radius + BLOCK_X) */];

        for (int cy = threadIdx.y; cy < 2 * radius + BLOCK_Y; cy += BLOCK_Y) {
            int sy = min(max(cy - static_cast<int>(threadIdx.y) - radius + y, 0), height - 1);
            for (int cx = threadIdx.x; cx < 2 * radius + BLOCK_X; cx += BLOCK_X) {
                int sx = min(max(cx - static_cast<int>(threadIdx.x) - radius + x, 0), width - 1);
                __pipeline_memcpy_async(
                    &buffer[cy * (2 * radius + BLOCK_X) + cx],
                    &src[sy * stride + sx],
                    4
                );
            }
        }

        if constexpr (has_ref) {
            for (int cy = threadIdx.y; cy < 2 * radius + BLOCK_Y; cy += BLOCK_Y) {
                int sy = min(max(cy - static_cast<int>(threadIdx.y) - radius + y, 0), height - 1);
                for (int cx = threadIdx.x; cx < 2 * radius + BLOCK_X; cx += BLOCK_X) {
                    int sx = min(max(cx - static_cast<int>(threadIdx.x) - radius + x, 0), width - 1);
                    __pipeline_memcpy_async(
                        &buffer[(2 * radius + BLOCK_Y + cy) * (2 * radius + BLOCK_X) + cx],
                        &src[(height + sy) * stride + sx],
                        4
                    );
                }
            }
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        if (x >= width || y >= height)
            return;

        const float center = buffer[
            (has_ref * (2 * radius + BLOCK_Y) + radius + threadIdx.y) * (2 * radius + BLOCK_X) +
            radius + threadIdx.x
        ]; // src[(has_ref * height + y) * stride + x];

        for (int cy = -radius; cy <= radius; ++cy) {
            int sy = cy + radius + threadIdx.y;

            for (int cx = -radius; cx <= radius; ++cx) {
                int sx = cx + radius + threadIdx.x;

                float value = buffer[(has_ref * (2 * radius + BLOCK_Y) + sy) * (2 * radius + BLOCK_X) + sx];

                float space = cy * cy + cx * cx;
                float range = (value - center) * (value - center);

                float weight = exp2f(space * sigma_spatial_scaled + range * sigma_color_scaled);

                if constexpr (has_ref) {
                    value = buffer[sy * (2 * radius + BLOCK_X) + sx];
                }

                num += weight * value;
                den += weight;
            }
        }
    } else {
        if (x >= width || y >= height)
            return;

        const float center = src[(has_ref * height + y) * stride + x];

        for (int cy = max(y - radius, 0); cy <= min(y + radius, height - 1); ++cy) {
            for (int cx = max(x - radius, 0); cx <= min(x + radius, width - 1); ++cx) {
                float value = src[(has_ref * height + cy) * stride + cx];

                float space = (y - cy) * (y - cy) + (x - cx) * (x - cx);
                float range = (value - center) * (value - center);

                float weight = exp2f(space * sigma_spatial_scaled + range * sigma_color_scaled);

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

cudaGraphExec_t get_graphexec(
    float * d_dst, float * d_src, float * h_buffer,
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    bool use_shared_memory, bool has_ref
) {

    size_t pitch { stride * sizeof(float) };

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t n_HtoD;
    {
        cudaMemcpy3DParms copy_params {};
        copy_params.srcPtr = make_cudaPitchedPtr(
            h_buffer, pitch, width, height * (1 + has_ref));
        copy_params.dstPtr = make_cudaPitchedPtr(
            d_src, pitch, width, height * (1 + has_ref));
        copy_params.extent = make_cudaExtent(
            width * sizeof(float), height * (1 + has_ref), 1);
        copy_params.kind = cudaMemcpyHostToDevice;

        cudaGraphAddMemcpyNode(&n_HtoD, graph, nullptr, 0, &copy_params);
    }

    cudaGraphNode_t n_kernel;
    {
        cudaGraphNode_t dependencies[] { n_HtoD };

        void * kernelArgs[] {
            &d_dst, &d_src,
            &width, &height, &stride,
            &sigma_spatial_scaled, &sigma_color_scaled, &radius
        };

        cudaKernelNodeParams kernel_params {};

        auto sharedMemBytes = static_cast<unsigned int>(
            (1 + has_ref) * (2 * radius + BLOCK_Y) * (2 * radius + BLOCK_X) * sizeof(float));
        bool useSharedMem = use_shared_memory && sharedMemBytes < 48 * 1024;

        kernel_params.func = (
            useSharedMem ?
            (has_ref ?
                reinterpret_cast<void *>(bilateral<true, true>) :
                reinterpret_cast<void *>(bilateral<true, false>)
            ) :
            (has_ref ?
                reinterpret_cast<void *>(bilateral<false, true>) :
                reinterpret_cast<void *>(bilateral<false, false>)
            )
        );
        kernel_params.blockDim = dim3(BLOCK_X, BLOCK_Y);
        kernel_params.gridDim = dim3(
            (width - 1) / BLOCK_X + 1,
            (height - 1) / BLOCK_Y + 1
        );
        kernel_params.sharedMemBytes = useSharedMem ? sharedMemBytes : 0;
        kernel_params.kernelParams = kernelArgs;

        cudaGraphAddKernelNode(
            &n_kernel, graph,
            dependencies, std::size(dependencies),
            &kernel_params);
    }

    cudaGraphNode_t n_DtoH;
    {
        cudaGraphNode_t dependencies[] { n_kernel };

        cudaMemcpy3DParms copy_params {};
        copy_params.srcPtr = make_cudaPitchedPtr(
            d_dst, pitch, width, height);
        copy_params.dstPtr = make_cudaPitchedPtr(
            h_buffer, pitch, width, height);
        copy_params.extent = make_cudaExtent(
            width * sizeof(float), height, 1);
        copy_params.kind = cudaMemcpyDeviceToHost;

        cudaGraphAddMemcpyNode(
            &n_DtoH, graph,
            dependencies, std::size(dependencies),
            &copy_params);
    }

    cudaGraphExec_t graphexecp;
    cudaGraphInstantiate(&graphexecp, graph, nullptr, nullptr, 0);

    cudaGraphDestroy(graph);

    return graphexecp;
}

