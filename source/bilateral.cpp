#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "cuda_runtime.h"

#include "vapoursynth/vapoursynth.h"
#include "vapoursynth/VSHelper.h"

using namespace std::string_literals;

extern void kernel(
    float * d_dst, const float * d_src, int width, int height, int stride, 
    float sigma_spatial, float sigma_color, int radius, cudaStream_t stream);

#define checkError(expr) do {                                                               \
    cudaError_t __err = expr;                                                               \
    if (__err != cudaSuccess) {                                                             \
        return set_error("'"s + # expr + "' failed: " + cudaGetErrorString(__err));         \
    }                                                                                       \
} while(0)

struct BilateralData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    float sigma_spatial[3], sigma_color[3];
    int radius[3], device_id;
    bool process[3] { true, true, true };

    int d_pitch;
    float * d_src = nullptr;
    float * d_dst = nullptr;
    float * h_buffer = nullptr;
    cudaStream_t stream;
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

        float * d_src = d->d_src;
        float * d_dst = d->d_dst;
        int d_pitch = d->d_pitch;
        int d_stride = d_pitch / sizeof(float);
        float * h_buffer = d->h_buffer;
        cudaStream_t stream = d->stream;

        int bps = d->vi->format->bitsPerSample;

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

                cudaMemcpy2DAsync(
                    d_src, d_pitch, h_buffer, d_pitch, width_bytes, height, 
                    cudaMemcpyHostToDevice, stream);

                kernel(
                    d_dst, d_src, width, height, d_stride, d->sigma_spatial[plane], 
                    d->sigma_color[plane], d->radius[plane], stream);

                cudaMemcpy2DAsync(
                    h_buffer, d_pitch, d_dst, d_pitch, width_bytes, height, 
                    cudaMemcpyDeviceToHost, stream);

                cudaError_t cudaResult = cudaStreamSynchronize(stream);
                if (cudaResult != cudaSuccess) {
                    vsapi->setFilterError(
                        ("BilateralGPU: "s + cudaGetErrorString(cudaResult)).c_str(), 
                        frameCtx
                    );

                    return nullptr;
                }

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

        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}

static void VS_CC BilateralFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    cudaFree(d->d_src);
    cudaFree(d->d_dst);
    cudaFreeHost(d->h_buffer);

    vsapi->freeNode(d->node);

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
        (sample == stFloat && bps != 32)) {

        return set_error("only constant format 8/16bit int or 32bit float input supported");
    }

    int error;

    for (int i = 0; i < std::extent_v<decltype(d->sigma_spatial)>; ++i) {
        float sigma_spatial = static_cast<float>(
            vsapi->propGetFloat(in, "sigma_spatial", i, &error));

        if (error) {
            if (i == 0) {
                sigma_spatial = 3.f;
            } else if (i == 1) {
                auto subH = d->vi->format->subSamplingH;
                auto subW = d->vi->format->subSamplingW;
                sigma_spatial = d->sigma_spatial[0] / std::sqrtf((1 << subH) * (1 << subW));
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

    for (int i = 0; i < std::extent_v<decltype(d->sigma_color)>; ++i) {
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

    for (int i = 0; i < std::extent_v<decltype(d->radius)>; ++i) {
        int radius = int64ToIntS(
            vsapi->propGetInt(in, "radius", i, &error));

        if (error) {
            radius = std::max(1, static_cast<int>(std::roundf(d->sigma_spatial[i] * 3.f)));
        } else if (radius <= 0) {
            return set_error("\"radius\" must be positive");
        }

        d->radius[i] = radius;
    }

    for (int i = 0; i < std::extent_v<decltype(d->sigma_spatial)>; ++i) {
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

    int width = d->vi->width;
    int height = d->vi->height;

    size_t d_pitch;
    checkError(cudaMallocPitch(
        &d->d_src, &d_pitch, width * sizeof(float), height));
    d->d_pitch = d_pitch;

    checkError(cudaMalloc(&d->d_dst, d_pitch * height * sizeof(float)));
    checkError(cudaHostAlloc(
        &d->h_buffer, d_pitch * height * sizeof(float), cudaHostAllocDefault));
    checkError(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));

    vsapi->createFilter(
        in, out, "Bilateral", 
        BilateralInit, BilateralGetFrame, BilateralFree, 
        fmParallelRequests, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {

    configFunc(
        "com.wolframrhodium.bilateralGPU", "bilateralgpu", "Bilateral filter using CUDA", 
        VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("Bilateral", 
        "clip:clip;"
        "sigma_spatial:float[]:opt;"
        "sigma_color:float[]:opt;"
        "radius:int[]:opt;"
        "device_id:int:opt",
        BilateralCreate, nullptr, plugin);
}
