#include <array>
#include <atomic>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <iterator>
#include <memory>
#include <mutex>
#include <numbers>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#ifdef _WIN64
#   include <windows.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

#include "kernel.hpp"

#ifdef _MSC_VER
#   if defined (_WINDEF_) && defined(min) && defined(max)
#       undef min
#       undef max
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#endif

using namespace std::string_literals;

#define checkError(expr) do {                                                         \
    if (CUresult result = expr; result != CUDA_SUCCESS) [[unlikely]] {                \
        const char * error_str;                                                       \
        cuGetErrorString(result, &error_str);                                         \
        return set_error("'"s + # expr + "' failed: " + error_str);                   \
    }                                                                                 \
} while(0)

#define checkNVRTCError(expr) do {                                                    \
    if (nvrtcResult result = expr; result != NVRTC_SUCCESS) [[unlikely]] {            \
        return set_error("'"s + # expr + "' failed: " + nvrtcGetErrorString(result)); \
    }                                                                                 \
} while(0)

struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

template <typename T, auto deleter>
    requires 
        std::default_initializable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T>
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;

    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}

    [[nodiscard]] constexpr Resource(Resource&& other) noexcept 
        : data(std::exchange(other.data, T{})) 
    { }

    constexpr Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
        }
    }

    constexpr Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

struct CUDA_Resource {
    Resource<CUdeviceptr, cuMemFree> d_src;
    Resource<CUdeviceptr, cuMemFree> d_dst;
    Resource<float *, cuMemFreeHost> h_buffer;
    Resource<CUstream, cuStreamDestroy> stream;
    std::array<Resource<CUgraphExec, cuGraphExecDestroy>, 3> graphexecs;
};

struct BilateralData {
    VSNodeRef * node;
    const VSVideoInfo * vi;

    // stored in graphexec
    // float sigma_spatial[3], sigma_color[3];
    // int radius[3];

    int device_id, num_streams;
    bool process[3] { true, true, true };

    int d_pitch;

    CUdevice device;
    CUcontext context; // use primary context
    ticket_semaphore semaphore;
    Resource<CUmodule, cuModuleUnload> modules[3];
    std::vector<CUDA_Resource> resources;
    std::mutex resources_lock;
};

static std::variant<CUmodule, std::string> compile(
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    bool use_shared_memory, int block_x, int block_y,
    CUdevice device
) noexcept {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    std::ostringstream kernel_source_io;
    kernel_source_io
        << std::hexfloat << std::boolalpha
        << "__device__ static const int width = " << width << ";\n"
        << "__device__ static const int height = " << height << ";\n"
        << "__device__ static const int stride = " << stride << ";\n"
        << "__device__ static const float sigma_spatial_scaled = " << sigma_spatial_scaled << ";\n"
        << "__device__ static const float sigma_color_scaled = " << sigma_color_scaled << ";\n"
        << "__device__ static const int radius = " << radius << ";\n"
        << "__device__ static const bool use_shared_memory = " << use_shared_memory << ";\n"
        << "#define BLOCK_X " << block_x << "\n"
        << "#define BLOCK_Y " << block_y << "\n"
        << kernel_source_template;
    const std::string kernel_source = kernel_source_io.str();

    nvrtcProgram program;
    checkNVRTCError(nvrtcCreateProgram(
        &program, kernel_source.c_str(), nullptr, 0, nullptr, nullptr));

    int major;
    checkError(cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    int minor;
    checkError(cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    int compute_capability = major * 10 + minor;

    // find maximum supported architecture
    int num_archs;
    checkNVRTCError(nvrtcGetNumSupportedArchs(&num_archs));
    const auto supported_archs = std::make_unique<int []>(num_archs);
    checkNVRTCError(nvrtcGetSupportedArchs(supported_archs.get()));
    bool generate_cubin = compute_capability <= supported_archs[num_archs - 1];

    const std::string arch_str = { 
        generate_cubin ? 
        "-arch=sm_" + std::to_string(compute_capability) : 
        "-arch=compute_" + std::to_string(supported_archs[num_archs - 1])
    };

    const char * opts[] = { 
        arch_str.c_str(), 
        "-use_fast_math", 
        "-std=c++17", 
        "-modify-stack-limit=false"
    };

    if (nvrtcCompileProgram(program, int{std::ssize(opts)}, opts) != NVRTC_SUCCESS) {
        size_t log_size;
        checkNVRTCError(nvrtcGetProgramLogSize(program, &log_size));
        std::string error_message;
        error_message.resize(log_size);
        checkNVRTCError(nvrtcGetProgramLog(program, error_message.data()));
        return set_error(error_message);
    }

    std::unique_ptr<char[]> image;
    if (generate_cubin) {
        size_t cubin_size;
        checkNVRTCError(nvrtcGetCUBINSize(program, &cubin_size));
        image = std::make_unique<char[]>(cubin_size);
        checkNVRTCError(nvrtcGetCUBIN(program, image.get()));
    } else {
        size_t ptx_size;
        checkNVRTCError(nvrtcGetPTXSize(program, &ptx_size));
        image = std::make_unique<char[]>(ptx_size);
        checkNVRTCError(nvrtcGetPTX(program, image.get()));
    }

    checkNVRTCError(nvrtcDestroyProgram(&program));

    CUmodule module_;
    checkError(cuModuleLoadData(&module_, image.get()));

    return module_;
}

static std::variant<CUgraphExec, std::string> get_graphexec(
    CUdeviceptr d_dst, CUdeviceptr d_src, float * h_buffer, 
    int width, int height, int stride, 
    int radius, 
    bool use_shared_memory, int block_x, int block_y, 
    CUcontext context, CUfunction function
) {

    const auto set_error = [](const std::string & error_message) {
        return error_message;
    };

    size_t pitch { stride * sizeof(float) };

    Resource<CUgraph, cuGraphDestroy> graph {};
    checkError(cuGraphCreate(&graph.data, 0));

    CUgraphNode n_HtoD;
    {
        CUDA_MEMCPY3D copy_params {
            .srcMemoryType = CU_MEMORYTYPE_HOST, 
            .srcHost = h_buffer, 
            .srcPitch = pitch, 
            .dstMemoryType = CU_MEMORYTYPE_DEVICE, 
            .dstDevice = d_src, 
            .dstPitch = pitch, 
            .WidthInBytes = width * sizeof(float), 
            .Height = static_cast<size_t>(height), 
            .Depth = 1, 
        };

        checkError(cuGraphAddMemcpyNode(
            &n_HtoD, graph, nullptr, 0, &copy_params, context));
    }

    CUgraphNode n_kernel;
    {
        CUgraphNode dependencies[] { n_HtoD };

        void * kernelParams[] { &d_dst, &d_src };

        unsigned int sharedMemBytes = (
            use_shared_memory ? 
            (2 * radius + block_y) * (2 * radius + block_x) * sizeof(float) : 
            0
        );
        
        CUDA_KERNEL_NODE_PARAMS node_params {
            .func = function, 
            .gridDimX = static_cast<unsigned int>((width - 1) / block_x + 1), 
            .gridDimY = static_cast<unsigned int>((height - 1) / block_y + 1), 
            .gridDimZ = 1, 
            .blockDimX = static_cast<unsigned int>(block_x), 
            .blockDimY = static_cast<unsigned int>(block_y), 
            .blockDimZ = 1, 
            .sharedMemBytes = sharedMemBytes, 
            .kernelParams = kernelParams, 
        };

        checkError(cuGraphAddKernelNode(
            &n_kernel, graph, dependencies, std::size(dependencies), &node_params));
    }

    CUgraphNode n_DtoH;
    {
        CUgraphNode dependencies[] { n_kernel };

        CUDA_MEMCPY3D copy_params {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE, 
            .srcDevice = d_dst, 
            .srcPitch = pitch, 
            .dstMemoryType = CU_MEMORYTYPE_HOST, 
            .dstHost = h_buffer, 
            .dstPitch = pitch, 
            .WidthInBytes = width * sizeof(float), 
            .Height = static_cast<size_t>(height), 
            .Depth = 1, 
        };

        checkError(cuGraphAddMemcpyNode(
            &n_DtoH, graph, 
            dependencies, std::size(dependencies), 
            &copy_params, context));
    }

    CUgraphExec graphexec;
    checkError(cuGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));

    return graphexec;
}


static void VS_CC BilateralInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node, 
    VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC BilateralGetFrame(
    int n, int activationReason, void **instanceData, void **frameData, 
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);

        const int pl[] = { 0, 1, 2 };
        const VSFrameRef * fr[] = { 
            d->process[0] ? nullptr : src, 
            d->process[1] ? nullptr : src, 
            d->process[2] ? nullptr : src 
        };

        VSFrameRef * dst = vsapi->newVideoFrame2(
            d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        d->semaphore.acquire();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        auto set_error = [&](const std::string & error_message) {
            d->resources_lock.lock();
            d->resources.push_back(std::move(resource));
            d->resources_lock.unlock();
            d->semaphore.release();
            vsapi->setFilterError(("BilateralGPU: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(src);
            return nullptr;
        };

        float * h_buffer = resource.h_buffer;
        CUstream stream = resource.stream;
        const auto & graphexecs = resource.graphexecs;

        checkError(cuCtxPushCurrent(d->context));

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!d->process[plane]) {
                continue;
            }

            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            int s_pitch = vsapi->getStride(src, plane);
            int bps = d->vi->format->bitsPerSample;
            int s_stride = s_pitch / (bps / 8);
            int width_bytes = width * sizeof(float);
            auto srcp = vsapi->getReadPtr(src, plane);
            int d_pitch = d->d_pitch;
            int d_stride = d_pitch / sizeof(float);

            if (bps == 32) {
                vs_bitblt(h_buffer, d_pitch, srcp, s_pitch, width_bytes, height);
            } else if (bps == 16) {
                float * h_bufferp = h_buffer;
                const uint16_t * src16p = reinterpret_cast<const uint16_t *>(srcp);

                for (int y = 0; y < height; ++y) {
#ifdef __AVX2__
                    // VideoFrame is at least 32 bytes padded
                    for (int x = 0; x < width; x += 8) {
                        __m128i src = _mm_load_si128(
                            reinterpret_cast<const __m128i *>(&src16p[x]));
                        __m256 srcf = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(src));
                        srcf = _mm256_mul_ps(srcf, 
                            _mm256_set1_ps(static_cast<float>(1.0 / 65535.0)));
                        _mm256_stream_ps(&h_bufferp[x], srcf);
                    }
#else
                    for (int x = 0; x < width; ++x) {
                        h_bufferp[x] = static_cast<float>(src16p[x]) / 65535.0f;
                    }
#endif

                    h_bufferp += d_stride;
                    src16p += s_stride;
                }
            } else if (bps == 8) {
                float * h_bufferp = h_buffer;
                const uint8_t * src8p = srcp;

                for (int y = 0; y < height; ++y) {
#ifdef __AVX2__
                    // VideoFrame is at least 32 bytes padded
                    for (int x = 0; x < width; x += 16) {
                        __m128i src = _mm_load_si128(
                            reinterpret_cast<const __m128i *>(&src8p[x]));
                        __m256 srcf_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(src));
                        srcf_lo = _mm256_mul_ps(
                            srcf_lo, 
                            _mm256_set1_ps(static_cast<float>(1.0 / 255.0)));
                        _mm256_stream_ps(&h_bufferp[x], srcf_lo);

                        __m256 srcf_hi = _mm256_cvtepi32_ps(
                            _mm256_cvtepu8_epi32(
                                _mm_castps_si128(
                                    _mm_permute_ps(
                                        _mm_castsi128_ps(src), 
                                        0b01'00'11'10))));
                        srcf_hi = _mm256_mul_ps(srcf_hi, 
                            _mm256_set1_ps(static_cast<float>(1.0 / 255.0)));
                        _mm256_stream_ps(&h_bufferp[x + 8], srcf_hi);
                    }
#else
                    for (int x = 0; x < width; ++x) {
                        h_bufferp[x] = static_cast<float>(src8p[x]) / 255.f;
                    }
#endif

                    h_bufferp += d_stride;
                    src8p += s_stride;
                }
            }

            checkError(cuGraphLaunch(graphexecs[plane], stream));
            checkError(cuStreamSynchronize(stream));

            auto dstp = vsapi->getWritePtr(dst, plane);

            if (bps == 32) {
                vs_bitblt(dstp, s_pitch, h_buffer, d_pitch, width_bytes, height);
            } else if (bps == 16) {
                uint16_t * dst16p = reinterpret_cast<uint16_t *>(dstp);
                const float * h_bufferp = h_buffer;

                for (int y = 0; y < height; ++y) {
#ifdef __AVX2__
                    // VideoFrame is at least 32 bytes padded
                    for (int x = 0; x < width; x += 8) {
                        __m256 dstf = _mm256_load_ps(&h_bufferp[x]);
                        dstf = _mm256_mul_ps(dstf, _mm256_set1_ps(65535.0f));
                        // dstf = _mm256_max_ps(dstf, _mm256_set1_ps(0.f));
                        // dstf = _mm256_min_ps(dstf, _mm256_set1_ps(65535.0f));
                        dstf = _mm256_round_ps(dstf, 
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        __m256i dsti32 = _mm256_cvtps_epi32(dstf);
                        __m128i dstu16 = _mm_packus_epi32(
                            _mm256_castsi256_si128(dsti32), 
                            _mm256_extractf128_si256(dsti32, 1)
                        );
                        _mm_stream_si128(reinterpret_cast<__m128i *>(&dst16p[x]), dstu16);
                    }
#else
                    for (int x = 0; x < width; ++x) {
                        float dstf = h_bufferp[x] * 65535.0f;
                        // dstf = std::clamp(dstf, 0.0f, 65535.0f);
                        dst16p[x] = static_cast<uint16_t>(std::roundf(dstf));
                    }
#endif

                    dst16p += s_stride;
                    h_bufferp += d_stride;
                }
            } else if (bps == 8) {
                uint8_t * dst8p = dstp;
                const float * h_bufferp = h_buffer;

                for (int y = 0; y < height; ++y) {
#ifdef __AVX2__
                    for (int x = 0; x < width; x += 8) {
                        __m256 dstf = _mm256_load_ps(&h_bufferp[x]);
                        dstf = _mm256_mul_ps(dstf, _mm256_set1_ps(255.0f));
                        // dstf = _mm256_max_ps(dstf, _mm256_set1_ps(0.f));
                        // dstf = _mm256_min_ps(dstf, _mm256_set1_ps(255.0f));
                        dstf = _mm256_round_ps(dstf, 
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        __m256i dsti32 = _mm256_cvtps_epi32(dstf);
                        __m128i dstu16 = _mm_packus_epi16(
                            _mm256_castsi256_si128(dsti32), 
                            _mm256_extractf128_si256(dsti32, 1)
                        );
                        __m128i dstu8 = _mm_shuffle_epi8(dstu16, _mm_setr_epi8(
                            0, 2, 4, 6, 8, 10, 12, 14, 
                            -1, -1, -1, -1, -1, -1, -1, -1));
                        *reinterpret_cast<long long *>(&dst8p[x]) = _mm_cvtsi128_si64(dstu8);
                    }
#else
                    for (int x = 0; x < width; ++x) {
                        float dstf = h_bufferp[x] * 255.0f;
                        // dstf = std::min(std::max(0.f, dstf), 255.0f);
                        dst8p[x] = static_cast<uint8_t>(std::roundf(dstf));
                    }
#endif

                    dst8p += s_stride;
                    h_bufferp += d_stride;
                }
            }
        }

        checkError(cuCtxPopCurrent(nullptr));

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}

static void VS_CC BilateralFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    vsapi->freeNode(d->node);

    auto device = d->device;

    cuCtxPushCurrent(d->context);

    delete d;

    cuCtxPopCurrent(nullptr);

    cuDevicePrimaryCtxRelease(device);
}

static void VS_CC BilateralCreate(
    const VSMap *in, VSMap *out, void *userData, 
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BilateralData>() };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("BilateralGPU: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    if (auto [bps, sample] = std::pair{ 
            d->vi->format->bitsPerSample, 
            d->vi->format->sampleType 
        };
        !isConstantFormat(d->vi) || 
        (sample == stInteger && (bps != 8 && bps != 16)) ||
        (sample == stFloat && bps != 32)
    ) {

        return set_error("only constant format 8/16bit int or 32bit float input supported");
    }

    int error;

    std::array<float, 3> sigma_spatial;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial[i] = static_cast<float>(
            vsapi->propGetFloat(in, "sigma_spatial", i, &error));

        if (error) {
            if (i == 0) {
                sigma_spatial[i] = 3.0f;
            } else if (i == 1) {
                auto subH = d->vi->format->subSamplingH;
                auto subW = d->vi->format->subSamplingW;
                sigma_spatial[i] = static_cast<float>(
                    sigma_spatial[0] / std::sqrt((1 << subH) * (1 << subW)));
            } else {
                sigma_spatial[i] = sigma_spatial[i - 1];
            }
        } else if (sigma_spatial[i] < 0.f) {
            return set_error("\"sigma_spatial\" must be non-negative");
        }

        if (sigma_spatial[i] < FLT_EPSILON) {
            d->process[i] = false;
        }
    }

    std::array<float, 3> sigma_spatial_scaled;
    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial_scaled[i] = (-0.5f / (sigma_spatial[i] * sigma_spatial[i])) * std::log2f(std::numbers::e_v<float>);
    }

    std::array<float, 3> sigma_color;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        sigma_color[i] = static_cast<float>(
            vsapi->propGetFloat(in, "sigma_color", i, &error));

        if (error) {
            if (i == 0) {
                sigma_color[i] = 0.02f;
            } else {
                sigma_color[i] = sigma_color[i - 1];
            }
        } else if (sigma_color[i] < 0.f) {
            return set_error("\"sigma_color\" must be non-negative");
        }
    }

    std::array<float, 3> sigma_color_scaled;
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            sigma_color_scaled[i] = (-0.5f / (sigma_color[i] * sigma_color[i])) * std::log2f(std::numbers::e_v<float>);
        }
    }

    std::array<int, 3> radius;
    for (int i = 0; i < std::ssize(radius); ++i) {
        radius[i] = int64ToIntS(vsapi->propGetInt(in, "radius", i, &error));

        if (error) {
            radius[i] = std::max(1, static_cast<int>(std::roundf(sigma_spatial[i] * 3.f)));
        } else if (radius[i] <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    // CUDA related
    {
        checkError(cuInit(0));

        int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
        if (error) {
            device_id = 0;
        }

        int device_count;
        checkError(cuDeviceGetCount(&device_count));
        if (0 <= device_id && device_id < device_count) {
            checkError(cuDeviceGet(&d->device, device_id));
        } else {
            return set_error("invalid device ID (" + std::to_string(device_id) + ")");
        }
        d->device_id = device_id;

        checkError(cuDevicePrimaryCtxRetain(&d->context, d->device));
        checkError(cuCtxPushCurrent(d->context));

        d->num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
        if (error) {
            d->num_streams = 4;
        }

        bool use_shared_memory = !!vsapi->propGetInt(in, "use_shared_memory", 0, &error);
        if (error) {
            use_shared_memory = true;
        }

        int block_x = int64ToIntS(vsapi->propGetInt(in, "block_x", 0, &error));
        if (error) {
            block_x = 16;
        }

        int block_y = int64ToIntS(vsapi->propGetInt(in, "block_y", 0, &error));
        if (error) {
            block_y = 8;
        }

        d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);

        d->resources.reserve(d->num_streams);

        int width = d->vi->width;
        int height = d->vi->height;
        int ssw = d->vi->format->subSamplingW;
        int ssh = d->vi->format->subSamplingH;

        int max_width { d->process[0] ? width : width >> ssw };
        int max_height { d->process[0] ? height : height >> ssh };

#ifdef _WIN64
        const std::string plugin_path = 
            vsapi->getPluginPath(vsapi->getPluginById("com.wolframrhodium.bilateralgpu_rtc", core));
        std::string folder_path = plugin_path.substr(0, plugin_path.find_last_of('/'));
        int nvrtc_major, nvrtc_minor;
        checkNVRTCError(nvrtcVersion(&nvrtc_major, &nvrtc_minor));
        const int nvrtc_version = nvrtc_major * 10 + nvrtc_minor;
        const std::string dll_path = 
            folder_path + "/nvrtc-builtins64_" + std::to_string(nvrtc_version) + ".dll";
        const Resource<HMODULE, FreeLibrary> dll_handle = LoadLibraryA(dll_path.c_str());
#endif

        CUfunction functions[3];
        for (int i = 0; i < d->num_streams; ++i) {
            Resource<CUdeviceptr, cuMemFree> d_src {};
            if (i == 0) {
                size_t d_pitch;
                checkError(cuMemAllocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float), max_height, 4));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cuMemAlloc(&d_src.data, max_height * d->d_pitch));
            }

            Resource<CUdeviceptr, cuMemFree> d_dst {};
            checkError(cuMemAlloc(&d_dst.data, max_height * d->d_pitch));

            Resource<float *, cuMemFreeHost> h_buffer {};
            checkError(cuMemAllocHost(
                reinterpret_cast<void **>(&h_buffer.data), max_height * d->d_pitch));

            Resource<CUstream, cuStreamDestroy> stream {};
            checkError(cuStreamCreate(&stream.data, CU_STREAM_NON_BLOCKING));

            std::array<Resource<CUgraphExec, cuGraphExecDestroy>, 3> graphexecs {};
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }

                if (i == 0) {
                    const auto result = compile(
                        width, height, d->d_pitch / sizeof(float),
                        sigma_spatial_scaled[plane], sigma_color_scaled[plane], radius[plane],
                        use_shared_memory, block_x, block_y,
                        d->device
                    );

                    if (std::holds_alternative<CUmodule>(result)) {
                        d->modules[plane] = std::get<CUmodule>(result);
                    } else {
                        return set_error(std::get<std::string>(result));
                    }

                    checkError(cuModuleGetFunction(
                        &functions[plane], d->modules[plane], "bilateral"));
                }

                int plane_width { plane == 0 ? width : width >> ssw };
                int plane_height { plane == 0 ? height : height >> ssh };

                const auto result = get_graphexec(
                    d_dst, d_src, h_buffer, 
                    plane_width, plane_height, d->d_pitch / sizeof(float), 
                    radius[plane], 
                    use_shared_memory, block_x, block_y, 
                    d->context, functions[plane]
                );

                if (std::holds_alternative<CUgraphExec>(result)) {
                    graphexecs[plane] = std::get<CUgraphExec>(result);
                } else {
                    return set_error(std::get<std::string>(result));
                }
            }

            d->resources.push_back(CUDA_Resource{
                .d_src = std::move(d_src), 
                .d_dst = std::move(d_dst), 
                .h_buffer = std::move(h_buffer), 
                .stream = std::move(stream), 
                .graphexecs = std::move(graphexecs)
            });
        }
    }

    vsapi->createFilter(
        in, out, "Bilateral", 
        BilateralInit, BilateralGetFrame, BilateralFree, 
        fmParallel, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {

    configFunc(
        "com.wolframrhodium.bilateralgpu_rtc", "bilateralgpu_rtc", 
        "Bilateral filter using CUDA (NVRTC)", 
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Bilateral", 
        "clip:clip;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "use_shared_memory:int:opt;"
        "block_x:int:opt;"
        "block_y:int:opt;",
        BilateralCreate, nullptr, plugin);
}
