#include <cfloat>
#include <cmath>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <vapoursynth/VapourSynth.h>
#include <vapoursynth/VSHelper.h>

using namespace std::string_literals;

extern cudaGraphExec_t get_graphexec(
    float * d_dst, float * d_src, float * h_buffer, 
    int width, int height, int stride, 
    float sigma_spatial, float sigma_color, int radius);

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
    float sigma_spatial[3], sigma_color[3];
    int radius[3], device_id, num_streams;
    bool process[3] { true, true, true };

    size_t d_pitch;
    ticket_semaphore semaphore;
    std::unique_ptr<std::atomic_flag[]> locks;
    std::vector<CUDA_Resource> resources;
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

        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);

        int bps = d->vi->format->bitsPerSample;

        int lock_idx = 0;
        d->semaphore.acquire();
        for (int i = 0; i < d->num_streams; ++i) {
            if (!d->locks[i].test_and_set(std::memory_order::acquire)) {
                lock_idx = i;
                break;
            }
        }

        auto set_error = [&](const std::string & error_message) {
            d->locks[lock_idx].clear(std::memory_order::release);
            d->semaphore.release();
            vsapi->setFilterError(("BilateralGPU: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(src);
            return nullptr;
        };

        float * h_buffer = d->resources[lock_idx].h_buffer.data;
        cudaStream_t stream = d->resources[lock_idx].stream.data;

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (d->process[plane]) {
                int width = vsapi->getFrameWidth(src, plane);
                int height = vsapi->getFrameHeight(src, plane);

                int s_pitch = vsapi->getStride(src, plane);
                int s_stride = s_pitch / (bps / 8);
                int width_bytes = width * sizeof(float);
                auto srcp = vsapi->getReadPtr(src, plane);

                if (bps == 32) {
                    vs_bitblt(h_buffer, d_pitch, srcp, s_pitch, width_bytes, height);
                } else if (bps == 16) {
                    auto h_bufferp = h_buffer;
                    auto src16p = reinterpret_cast<const uint16_t *>(srcp);

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            h_bufferp[x] = static_cast<float>(src16p[x]) / 65535.f;
                        }

                        h_bufferp += d_stride;
                        src16p += s_stride;
                    }
                } else if (bps == 8) {
                    auto h_bufferp = h_buffer;
                    auto src8p = srcp;

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            h_bufferp[x] = static_cast<float>(src8p[x]) / 255.f;
                        }

                        h_bufferp += d_stride;
                        src8p += s_stride;
                    }
                }

                cudaGraphExec_t graphexec = d->resources[lock_idx].graphexecs[plane];
                checkError(cudaGraphLaunch(graphexec, stream));
                checkError(cudaStreamSynchronize(stream));

                auto dstp = vsapi->getWritePtr(dst, plane);

                if (bps == 32) {
                    vs_bitblt(dstp, s_pitch, h_buffer, d_pitch, width_bytes, height);
                } else if (bps == 16) {
                    auto dst16p = reinterpret_cast<uint16_t *>(dstp);
                    auto h_bufferp = h_buffer;

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            float dstf = h_bufferp[x] * 65535.f;
                            float clamped_dstf = std::min(std::max(0.f, dstf + 0.5f), 65535.f);
                            dst16p[x] = static_cast<uint16_t>(std::roundf(clamped_dstf));
                        }

                        dst16p += s_stride;
                        h_bufferp += d_stride;
                    }
                } else if (bps == 8) {
                    auto dst8p = dstp;
                    auto h_bufferp = h_buffer;

                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            float dstf = h_bufferp[x] * 255.f;
                            float clamped_dstf = std::min(std::max(0.f, dstf + 0.5f), 255.f);
                            dst8p[x] = static_cast<uint8_t>(std::roundf(clamped_dstf));
                        }

                        dst8p += s_stride;
                        h_bufferp += d_stride;
                    }
                }
            }
        }

        d->locks[lock_idx].clear(std::memory_order::release);
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

    for (int i = 0; i < std::ssize(d->sigma_spatial); ++i) {
        float sigma_spatial = static_cast<float>(
            vsapi->propGetFloat(in, "sigma_spatial", i, &error));

        if (error) {
            if (i == 0) {
                sigma_spatial = 3.0f;
            } else if (i == 1) {
                auto subH = d->vi->format->subSamplingH;
                auto subW = d->vi->format->subSamplingW;
                sigma_spatial = d->sigma_spatial[0] / std::sqrt((1 << subH) * (1 << subW));
            } else {
                sigma_spatial = d->sigma_spatial[i - 1];
            }
        } else if (sigma_spatial < 0.f) {
            return set_error("\"sigma_spatial\" must be non-negative");
        }

        d->sigma_spatial[i] = sigma_spatial; // unscaled before parsing argument "radius"

        if (sigma_spatial < FLT_EPSILON) {
            d->process[i] = false;
        }
    }

    for (int i = 0; i < std::ssize(d->sigma_color); ++i) {
        float sigma_color = static_cast<float>(
            vsapi->propGetFloat(in, "sigma_color", i, &error));

        if (error) {
            if (i == 0) {
                sigma_color = 0.02f;
            } else {
                sigma_color = d->sigma_color[i - 1];
            }
        } else if (sigma_color < 0.f) {
            return set_error("\"sigma_color\" must be non-negative");
        }

        d->sigma_color[i] = -0.5f / (sigma_color * sigma_color);

        if (sigma_color < FLT_EPSILON) {
            d->process[i] = false;
        }
    }

    for (int i = 0; i < std::ssize(d->radius); ++i) {
        int radius = int64ToIntS(
            vsapi->propGetInt(in, "radius", i, &error));

        if (error) {
            radius = std::max(1, static_cast<int>(std::roundf(d->sigma_spatial[i] * 3.f)));
        } else if (radius <= 0) {
            return set_error("\"radius\" must be positive");
        }

        d->radius[i] = radius;
    }

    for (int i = 0; i < std::ssize(d->sigma_spatial); ++i) {
        d->sigma_spatial[i] = -0.5f / (d->sigma_spatial[i] * d->sigma_spatial[i]);
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

    {
        d->semaphore.current.store(d->num_streams - 1, std::memory_order::relaxed);
        d->locks = std::make_unique<std::atomic_flag[]>(d->num_streams);

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
                checkError(cudaMallocPitch(
                    &d_src.data, &d->d_pitch, max_width * sizeof(float), max_height));
            } else {
                checkError(cudaMalloc(&d_src.data, max_height * d->d_pitch));
            }

            Resource<float *, cudaFree> d_dst {};
            checkError(cudaMalloc(&d_dst.data, max_height * d->d_pitch));

            Resource<float *, cudaFreeHost> h_buffer {};
            checkError(cudaMallocHost(&h_buffer.data, max_height * d->d_pitch));

            Resource<cudaStream_t, cudaStreamDestroy> stream {};
            checkError(cudaStreamCreateWithFlags(&stream.data, cudaStreamNonBlocking));

            std::array<Resource<cudaGraphExec_t, cudaGraphExecDestroy>, 3> graphexecs;
            for (int plane = 0; plane < d->vi->format->numPlanes; ++plane) {
                if (d->process[plane]) {
                    int plane_width { plane == 0 ? width : width >> ssw };
                    int plane_height { plane == 0 ? height : height >> ssh };

                    graphexecs[plane] = get_graphexec(
                        d_dst, d_src, h_buffer, 
                        plane_width, plane_height, d->d_pitch / sizeof(float), 
                        d->sigma_spatial[plane], d->sigma_color[plane], 
                        d->radius[plane]
                    );
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
        "com.wolframrhodium.bilateralgpu", "bilateralgpu", "Bilateral filter using CUDA", 
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Bilateral", 
        "clip:clip;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;",
        BilateralCreate, nullptr, plugin);
}