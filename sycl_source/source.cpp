#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

#include <config.h>

#ifndef USE_DEAFAULT_CONTEXT
    #if defined(SYCL_EXT_ONEAPI_DEFAULT_CONTEXT) && SYCL_EXT_ONEAPI_DEFAULT_CONTEXT >= 1
        #define USE_DEFAULT_CONTEXT 1
    #else
        #define USE_DEFAULT_CONTEXT 0
    #endif
#endif

extern sycl::event launch(
    float * d_dst, float * d_src, float * h_buffer,
    int width, int height, int stride,
    float sigma_spatial_scaled, float sigma_color_scaled, int radius,
    bool use_shared_memory,
    int block_x, int block_y,
    bool has_ref,
    sycl::queue & stream
);

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

struct SYCL_Resource {
    float * d_src;
    float * d_dst;
    float * h_buffer;
    std::unique_ptr<sycl::queue> stream;
};

struct BilateralData {
    VSNodeRef * node;
    VSNodeRef * ref_node;
    const VSVideoInfo * vi;

    float sigma_spatial_scaled[3], sigma_color_scaled[3];
    int radius[3];
    bool use_shared_memory;
    int block_x, block_y;

    int num_streams;
    std::unique_ptr<sycl::device> device;
    std::unique_ptr<sycl::context> context;
    bool process[3] { true, true, true };

    int d_pitch;
    ticket_semaphore semaphore;
    std::vector<SYCL_Resource> resources;
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
        if (d->ref_node) {
            vsapi->requestFrameFilter(n, d->ref_node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * ref = nullptr;
        if (d->ref_node) {
            ref = vsapi->getFrameFilter(n, d->ref_node, frameCtx);
        }

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
            vsapi->setFilterError(("BilateralSYCL: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            if (d->ref_node) {
                vsapi->freeFrame(ref);
            }
            vsapi->freeFrame(src);
            return nullptr;
        };

        float * h_buffer = resource.h_buffer;

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

            const uint8_t * refp = nullptr;
            if (d->ref_node) {
                refp = vsapi->getReadPtr(ref, plane);
            }

            if (bps == 32) {
                vs_bitblt(h_buffer, d_pitch, srcp, s_pitch, width_bytes, height);
                if (d->ref_node) {
                    vs_bitblt(&h_buffer[s_stride * height], d_pitch, refp, s_pitch, width_bytes, height);
                }
            } else if (bps == 16) {
                float * h_bufferp = h_buffer;

                const auto load = [width, height, &h_bufferp, s_stride, d_stride](const uint16_t * srcp) {
                    for (int y = 0; y < height; ++y) {
#ifdef __AVX2__
                        // VideoFrame is at least 32 bytes padded
                        for (int x = 0; x < width; x += 8) {
                            __m128i src = _mm_load_si128(
                                reinterpret_cast<const __m128i *>(&srcp[x]));
                            __m256 srcf = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(src));
                            srcf = _mm256_mul_ps(srcf,
                                _mm256_set1_ps(static_cast<float>(1.0 / 65535.0)));
                            _mm256_stream_ps(&h_bufferp[x], srcf);
                        }
#else
                        for (int x = 0; x < width; ++x) {
                            h_bufferp[x] = static_cast<float>(srcp[x]) / 65535.0f;
                        }
#endif

                        h_bufferp += d_stride;
                        srcp += s_stride;
                    }
                };

                load(reinterpret_cast<const uint16_t *>(srcp));
                if (d->ref_node) {
                    load(reinterpret_cast<const uint16_t *>(refp));
                }
            } else if (bps == 8) {
                float * h_bufferp = h_buffer;

                const auto load = [width, height, &h_bufferp, s_stride, d_stride](const uint8_t * srcp) {
                    for (int y = 0; y < height; ++y) {
#ifdef __AVX2__
                    // VideoFrame is at least 32 bytes padded
                        for (int x = 0; x < width; x += 16) {
                            __m128i src = _mm_load_si128(
                                reinterpret_cast<const __m128i *>(&srcp[x]));
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
                            h_bufferp[x] = static_cast<float>(srcp[x]) / 255.f;
                        }
#endif

                        h_bufferp += d_stride;
                        srcp += s_stride;
                    }
                };

                load(srcp);
                if (d->ref_node) {
                    load(reinterpret_cast<const uint8_t *>(refp));
                }
            }

            try{
                launch(
                    resource.d_dst, resource.d_src, resource.h_buffer,
                    width, height, d_stride,
                    d->sigma_spatial_scaled[plane], d->sigma_color_scaled[plane], d->radius[plane],
                    d->use_shared_memory,
                    d->block_x, d->block_y,
                    d->ref_node != nullptr,
                    *resource.stream
                ).wait();
            } catch (const std::exception & e) {
                return set_error(e.what());
            }

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

        if (d->ref_node) {
            vsapi->freeFrame(ref);
        }
        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}

static void VS_CC BilateralFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }
    vsapi->freeNode(d->node);

    for (const auto & resource : d->resources) {
        sycl::free(resource.h_buffer, *d->context);
        sycl::free(resource.d_dst, *d->context);
        sycl::free(resource.d_src, *d->context);
    }

    delete d;
}

static void VS_CC BilateralCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi) {

    auto d { std::make_unique<BilateralData>() };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    int error;

    d->ref_node = vsapi->propGetNode(in, "ref", 0, &error);
    bool has_ref = d->ref_node != nullptr;

    auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("BilateralGPU: " + error_message).c_str());
        if (has_ref) {
            vsapi->freeNode(d->ref_node);
        }
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

    const auto ref_vi = vsapi->getVideoInfo(d->ref_node);
    if (d->ref_node && (!isSameFormat(d->vi, ref_vi) || d->vi->numFrames != ref_vi->numFrames)) {
        return set_error("\"ref\" must be of the same format as \"clip\"");
    }

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

    for (int i = 0; i < std::ssize(sigma_spatial); ++i) {
        d->sigma_spatial_scaled[i] = -0.5f / (sigma_spatial[i] * sigma_spatial[i]) * std::numbers::log2e_v<float>;
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

    for (int i = 0; i < std::ssize(sigma_color); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            d->sigma_color_scaled[i] = (-0.5f / (sigma_color[i] * sigma_color[i])) * std::numbers::log2e_v<float>;
        }
    }

    for (int i = 0; i < std::ssize(d->radius); ++i) {
        d->radius[i] = int64ToIntS(vsapi->propGetInt(in, "radius", i, &error));

        if (error) {
           d->radius[i] = std::max(1, static_cast<int>(std::roundf(sigma_spatial[i] * 3.f)));
        } else if (d->radius[i] <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    std::vector<sycl::device> devices;
    for (const auto & platform : sycl::platform::get_platforms()) {
        for (const auto & device : platform.get_devices()) {
            devices.emplace_back(device);
        }
    }
    if (0 <= device_id && device_id < static_cast<int>(devices.size())) {
        d->device = std::make_unique<sycl::device>(devices[device_id]);
    } else {
        return set_error("invalid \"device_id\"");
    }

    #if USE_DEFAULT_CONTEXT
    try {
        d->context = std::make_unique<sycl::context>(d->device->get_platform().ext_oneapi_get_default_context());
    } catch (const std::runtime_error & e) {
    #endif
        d->context = std::make_unique<sycl::context>(*d->device);
    #if USE_DEFAULT_CONTEXT
    }
    #endif

    d->num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        d->num_streams = 4;
    }

    d->use_shared_memory = !!vsapi->propGetInt(in, "use_shared_memory", 0, &error);
    if (error) {
        d->use_shared_memory = true;
    }

    d->block_x = int64ToIntS(vsapi->propGetInt(in, "block_x", 0, &error));
    if (error) {
        d->block_x = 16;
    }

    d->block_y = int64ToIntS(vsapi->propGetInt(in, "block_y", 0, &error));
    if (error) {
        d->block_y = 16;
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

        d->d_pitch = max_width * sizeof(float);

        for (int i = 0; i < d->num_streams; ++i) {
            auto d_src = sycl::malloc_device<float>(
                (1 + (d->ref_node != nullptr)) * max_height * max_width,
                *d->device, *d->context
            );

            auto d_dst = sycl::malloc_device<float>(
                max_height * max_width,
                *d->device, *d->context
            );

            auto h_buffer = sycl::malloc_host<float>(
                (1 + (d->ref_node != nullptr)) * max_height * max_width,
                *d->context
            );

            auto stream = std::make_unique<sycl::queue>(*d->context, *d->device);

            d->resources.push_back(SYCL_Resource{
                .d_src = std::move(d_src),
                .d_dst = std::move(d_dst),
                .h_buffer = std::move(h_buffer),
                .stream = std::move(stream),
            });
        }
    }

    vsapi->createFilter(
        in, out, "Bilateral",
        BilateralInit, BilateralGetFrame, BilateralFree,
        fmParallel, 0, d.release(), core);
}

static void VS_CC DeviceInfo(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    int error;
    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    std::vector<sycl::device> devices;
    for (const auto & platform : sycl::platform::get_platforms()) {
        for (const auto & device : platform.get_devices()) {
            devices.emplace_back(device);
        }
    }
    if (0 <= device_id && device_id < static_cast<int>(devices.size())) {
        using namespace sycl::info;
        const auto & device = devices[device_id];

        {
            const auto & platform = device.get_info<device::platform>();
            vsapi->propSetData(out, "platform_name", platform.get_info<platform::name>().c_str(), -1, paReplace);
            vsapi->propSetData(out, "platform_version", platform.get_info<platform::version>().c_str(), -1, paReplace);
        }

        vsapi->propSetData(out, "name", device.get_info<device::name>().c_str(), -1, paReplace);
        vsapi->propSetInt(out, "is_available", device.get_info<device::is_available>(), paReplace);
        vsapi->propSetData(out, "vendor", device.get_info<device::vendor>().c_str(), -1, paReplace);
        vsapi->propSetData(out, "driver_version", device.get_info<device::driver_version>().c_str(), -1, paReplace);
        vsapi->propSetData(out, "version", device.get_info<device::version>().c_str(), -1, paReplace);

        try {
            const auto sub_group_sizes = device.get_info<device::sub_group_sizes>();
            vsapi->propSetIntArray(
                out, "sub_group_sizes",
                (const int64_t *) sub_group_sizes.data(),
                (int) sub_group_sizes.size()
            );
        } catch (const sycl::exception &) {
            vsapi->propSetData(out, "sub_group_size", "not supported", -1, paReplace);
        }

        vsapi->propSetInt(out, "max_compute_units", device.get_info<device::max_compute_units>(), paReplace);
        vsapi->propSetInt(out, "local_mem_size", device.get_info<device::local_mem_size>(), paReplace);
        vsapi->propSetInt(out, "error_correction_support", device.get_info<device::error_correction_support>(), paReplace);

        auto device_string = [](device_type type) {
            switch (type) {
                case device_type::cpu:
                    return "cpu";
                case device_type::gpu:
                    return "gpu";
                case device_type::accelerator:
                    return "accelerator";
                case device_type::custom:
                    return "custom";
                case device_type::host:
                    return "host";
                default:
                    return "unknown";
            }
        };
        vsapi->propSetData(
            out,
            "type",
            device_string(device.get_info<device::device_type>()),
            -1,
            paReplace
        );

        auto global_mem_cache_string = [](global_mem_cache_type type) {
            switch (type) {
                case global_mem_cache_type::none:
                    return "none";
                case global_mem_cache_type::read_only:
                    return "read_only";
                case global_mem_cache_type::read_write:
                    return "read_write";
                default:
                    return "unknown";
            }
        };
        vsapi->propSetData(
            out,
            "global_mem_cache_type",
            global_mem_cache_string(device.get_info<device::global_mem_cache_type>()),
            -1,
            paReplace
        );

        auto local_mem_string = [](local_mem_type type) {
            switch (type) {
                case local_mem_type::none:
                    return "none";
                case local_mem_type::local:
                    return "local";
                case local_mem_type::global:
                    return "global";
                default:
                    return "unknown";
            }
        };
        vsapi->propSetData(
            out,
            "local_mem_type",
            local_mem_string(device.get_info<device::local_mem_type>()),
            -1,
            paReplace
        );
    } else {
        vsapi->setError(out, "invalid \"device_id\"");
    }
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {

    configFunc(
        "com.wolframrhodium.bilateralsycl", "bilateralsycl", "Bilateral filter using SYCL",
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
        "block_y:int:opt;"
        "ref:clip:opt;",
        BilateralCreate, nullptr, plugin);

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);

    registerFunc("DeviceInfo", "device_id:int:opt", DeviceInfo, nullptr, plugin);
}
