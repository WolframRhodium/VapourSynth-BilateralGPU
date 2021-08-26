#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_dst, float * d_src, float * h_buffer, 
    int width, int height, int stride, 
    float sigma_spatial, float sigma_color, int radius, 
    bool use_shared_memory);

#define checkError(expr) do {                                                               \
    cudaError_t __err = expr;                                                               \
    if (__err != cudaSuccess) {                                                             \
        return set_error("'"s + # expr + "' failed: " + cudaGetErrorString(__err));         \
    }                                                                                       \
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
    Resource<float *, cudaFree> d_src;
    Resource<float *, cudaFree> d_dst;
    Resource<float *, cudaFreeHost> h_buffer;
    Resource<cudaStream_t, cudaStreamDestroy> stream;
    std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs;
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
    ticket_semaphore semaphore;
    std::vector<CUDA_Resource> resources;
    std::mutex resources_lock;
};

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
        cudaStream_t stream = resource.stream;
        const auto & graphexecs = resource.graphexecs;

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

            checkError(cudaGraphLaunch(graphexecs[plane], stream));
            checkError(cudaStreamSynchronize(stream));

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

    cudaSetDevice(d->device_id);

    delete d;
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

    float sigma_spatial[3];
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

    float sigma_color[3];
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
    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            sigma_color[i] = -0.5f / (sigma_color[i] * sigma_color[i]);
        }
    }

    int radius[3];
    for (int i = 0; i < std::ssize(radius); ++i) {
        radius[i] = int64ToIntS(vsapi->propGetInt(in, "radius", i, &error));

        if (error) {
            radius[i] = std::max(1, static_cast<int>(std::roundf(sigma_spatial[i] * 3.f)));
        } else if (radius[i] <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        sigma_spatial[i] = -0.5f / (sigma_spatial[i] * sigma_spatial[i]);
    }

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(cudaSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    d->num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 4;
    }

    bool use_shared_memory = !!vsapi->propGetInt(in, "use_shared_memory", 0, &error);
    if (error) {
        use_shared_memory = true;
    }

    {
        d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);

        d->resources.reserve(d->num_streams);

        int width = d->vi->width;
        int height = d->vi->height;
        int ssw = d->vi->format->subSamplingW;
        int ssh = d->vi->format->subSamplingH;

        int max_width { d->process[0] ? width : width >> ssw };
        int max_height { d->process[0] ? height : height >> ssh };

        for (int i = 0; i < d->num_streams; ++i) {
            Resource<float *, cudaFree> d_src {};
            if (i == 0) {
                size_t d_pitch;
                checkError(cudaMallocPitch(
                    &d_src.data, &d_pitch, max_width * sizeof(float), max_height));
                d->d_pitch = static_cast<int>(d_pitch);
            } else {
                checkError(cudaMalloc(&d_src.data, max_height * d->d_pitch));
            }

            Resource<float *, cudaFree> d_dst {};
            checkError(cudaMalloc(&d_dst.data, max_height * d->d_pitch));

            Resource<float *, cudaFreeHost> h_buffer {};
            checkError(cudaMallocHost(&h_buffer.data, max_height * d->d_pitch));

            Resource<cudaStream_t, cudaStreamDestroy> stream {};
            checkError(cudaStreamCreateWithFlags(&stream.data, cudaStreamNonBlocking));

            std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs {};
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                if (!d->process[plane]) {
                    continue;
                }

                int plane_width { plane == 0 ? width : width >> ssw };
                int plane_height { plane == 0 ? height : height >> ssh };

                graphexecs[plane] = get_graphexec(
                    d_dst, d_src, h_buffer, 
                    plane_width, plane_height, d->d_pitch / sizeof(float), 
                    sigma_spatial[plane], sigma_color[plane], radius[plane], 
                    use_shared_memory
                );
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
        "com.wolframrhodium.bilateralgpu", "bilateralgpu", "Bilateral filter using CUDA", 
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Bilateral", 
        "clip:clip;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "use_shared_memory:int:opt;",
        BilateralCreate, nullptr, plugin);
}
