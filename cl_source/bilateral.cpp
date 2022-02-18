#include <array>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <ios>
#include <iostream>
#include <memory>
#include <numbers>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if !defined(__cpp_lib_atomic_wait)
#  include <chrono>
#endif // !defined(__cpp_lib_atomic_wait)

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <config.h>
#include "kernel.hpp"

#define checkError(expr) do {                                \
    if (cl_int __result = expr; __result != CL_SUCCESS) {    \
        std::cerr << # expr << ' ' << __result << std::endl; \
        abort();                                             \
    }                                                        \
} while(0)

#define CALL(F, ...) [&](){                                  \
    cl_int __err;                                            \
    auto __ret = F(__VA_ARGS__, &__err);                     \
    if (__err != CL_SUCCESS) {                               \
        std::cerr << "error" << __err << std::endl;          \
        abort();                                             \
    }                                                        \
    return __ret;                                            \
}()

namespace {
    constexpr auto max_num_planes = 3;
}


struct ticket_semaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order_acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order_acquire) };
            if (tk <= curr) {
                return;
            }
#if defined(__cpp_lib_atomic_wait)
            current.wait(curr, std::memory_order_relaxed);
#else // defined(__cpp_lib_atomic_wait)
            auto duration = std::chrono::milliseconds(1);
            std::this_thread::sleep_for(duration);
#endif // defined(__cpp_lib_atomic_wait)
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
        current.notify_all();
#endif // defined(__cpp_lib_atomic_wait)
    }
};


struct Params {
    int block_x;
    int block_y;
    int width;
    int height;
    float sigma_spatial_scaled;
    float sigma_color_scaled;
    int radius;
    bool use_shared_memory;
};


static inline constexpr int get_stride(int width) noexcept {
    constexpr int align = 1 << 5;
    return (width + (align - 1)) / align * align;
}


struct Resource {
private:
    cl_mem h_src;
    void * h_src_ptr;
    cl_mem d_src;
    cl_mem d_dst;
    cl_mem h_dst;
    void * h_dst_ptr;
    std::array<Params, max_num_planes> params;

public:
    Resource(
        cl_context context,
        cl_device_id device,
        const std::array<Params, max_num_planes> & params,
        cl_command_queue command_queue
    ) noexcept {

        this->params = params;

        this->h_src = this->init_buffer(
            context, this->params[0].width, this->params[0].height,
            CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR);

        this->h_src_ptr = this->map_pointer(
            command_queue, this->h_src, this->params[0].width, this->params[0].height,
            CL_MAP_WRITE_INVALIDATE_REGION);

        this->d_src = this->init_buffer(
            context, this->params[0].width, this->params[0].height,
            CL_MEM_READ_ONLY);

        this->d_dst = this->init_buffer(
            context, this->params[0].width, this->params[0].height,
            CL_MEM_WRITE_ONLY);

        this->h_dst = this->init_buffer(
            context, this->params[0].width, this->params[0].height,
            CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);

        this->h_dst_ptr = this->map_pointer(
            command_queue, this->h_dst, this->params[0].width, this->params[0].height,
            CL_MAP_READ);
    }

private:
    static inline cl_mem init_buffer(
        cl_context context, int width, int height, cl_mem_flags flags = {}
    ) noexcept {
        return CALL(
            clCreateBuffer,
            context, flags, height * get_stride(width) * sizeof(float), nullptr);
    }

    static inline void * map_pointer(
        cl_command_queue command_queue, cl_mem buffer, int width, int height,
        cl_map_flags flags = {}
    ) noexcept {
        return CALL(
            clEnqueueMapBuffer,
            command_queue, buffer, CL_TRUE, flags, 0,
            height * get_stride(width) * sizeof(float), 0, nullptr, nullptr);
    }

public:
    inline void execute(
        float * dstp,
        cl_kernel kernel,
        const float * srcp,
        int stride,
        int plane,
        cl_command_queue command_queue
    ) noexcept {

        vs_bitblt(
            this->h_src_ptr, get_stride(this->params[plane].width) * sizeof(float),
            srcp, stride * sizeof(float),
            this->params[plane].width * sizeof(float),
            this->params[plane].height
        );

        checkError(clEnqueueWriteBuffer(
            command_queue, this->d_src, CL_FALSE, 0,
            this->params[plane].height * get_stride(this->params[plane].width) * sizeof(float),
            this->h_src_ptr, 0, nullptr, nullptr));

        checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->d_dst));
        checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->d_src));

        constexpr cl_uint work_dim = 2;
        const size_t global_work_size[work_dim] {
            (size_t) this->params[plane].width,
            (size_t) this->params[plane].height
        };
        const size_t local_work_size[work_dim] {
            (size_t) this->params[plane].block_x,
            (size_t) this->params[plane].block_y
        };
        checkError(clEnqueueNDRangeKernel(
            command_queue, kernel, work_dim, nullptr,
            global_work_size, local_work_size,
            0, nullptr, nullptr));

        checkError(clEnqueueReadBuffer(
            command_queue, d_dst, CL_FALSE, 0,
            this->params[plane].height * get_stride(this->params[plane].width) * sizeof(float),
            this->h_dst_ptr, 0, nullptr, nullptr));

        checkError(clFinish(command_queue));

        vs_bitblt(
            dstp, stride * sizeof(float),
            this->h_dst_ptr, get_stride(this->params[plane].width) * sizeof(float),
            this->params[plane].width * sizeof(float),
            this->params[plane].height
        );
    }

    inline void destroy() noexcept {
        checkError(clReleaseMemObject(this->h_dst));
        checkError(clReleaseMemObject(this->d_dst));
        checkError(clReleaseMemObject(this->d_src));
        checkError(clReleaseMemObject(this->h_src));
    }
};


struct ResourcePool {
private:
    cl_context context {};
    std::array<cl_program, 3> programs;
    std::vector<std::array<cl_kernel, 3>> kernels;
    std::vector<cl_command_queue> command_queues;

    std::vector<Resource> resources;

    ticket_semaphore semaphore;
    std::vector<int> tickets;
    std::atomic_flag tickets_lock;

public:
    inline void init(
        int device_id,
        const std::array<Params, 3> & params,
        int num_resources,
        const std::array<bool, 3> & process
    ) noexcept {

        std::vector<cl_platform_id> platforms { this->init_platforms() };

        auto [platform, device] = this->init_device(platforms, device_id);

        this->context = this->init_context(platform, device);

        this->kernels.resize(num_resources);

        for (unsigned plane = 0; plane < std::size(params); ++plane) {
            if (process[plane]) {
                this->programs[plane] = this->init_program(this->context, device, params[plane]);

                for (int i = 0; i < num_resources; ++i) {
                    this->kernels[i][plane] = this->init_kernel(this->programs[plane], "bilateral");
                }
            } else {
                this->programs[plane] = nullptr;

                for (int i = 0; i < num_resources; ++i) {
                    this->kernels[i][plane] = nullptr;
                }
            }
        }

        this->command_queues = this->init_command_queue(this->context, device, num_resources);

        this->resources.reserve(num_resources);
        for (int i = 0; i < num_resources; ++i) {
            this->resources.emplace_back(Resource{this->context, device, params, this->command_queues[0]});
        }

        this->semaphore.current.store(num_resources - 1, std::memory_order::relaxed);

        this->tickets.reserve(num_resources);
        for (int i = 0; i < num_resources; ++i) {
            this->tickets.emplace_back(i);
        }

        checkError(clReleaseDevice(device));
    }

private:
    static inline std::vector<cl_platform_id> init_platforms() noexcept {
        cl_uint count;
        checkError(clGetPlatformIDs(0, nullptr, &count));

        std::vector<cl_platform_id> platforms(count);
        checkError(clGetPlatformIDs(count, platforms.data(), nullptr));

        return platforms;
    }

    static inline std::tuple<cl_platform_id, cl_device_id> init_device(
        std::vector<cl_platform_id> platforms,
        int device_id
    ) noexcept {
        std::vector<std::tuple<cl_platform_id, cl_device_id>> devices;

        for (const auto platform : platforms) {
            cl_uint count;
            checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &count));

            for (cl_uint i = 0; i < count; ++i) {
                cl_device_id device;
                checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr));
                devices.emplace_back(platform, device);
            }
        }

        for (unsigned i = 0; i < devices.size(); ++i) {
            if (((int) i) != device_id) {
                checkError(clReleaseDevice(std::get<1>(devices[i])));
            }
        }

        return devices[device_id];
    }

    static inline cl_context init_context(
        cl_platform_id platform,
        cl_device_id device
    ) noexcept {
        return CALL(clCreateContext, nullptr, 1, &device, nullptr, nullptr);
    }

    static inline cl_program init_program(
        cl_context context,
        cl_device_id device,
        const Params & param
    ) noexcept {

        std::ostringstream program_stream;
        program_stream
            << "#define BLOCK_X " << param.block_x << '\n'
            << "#define BLOCK_Y " << param.block_y << '\n'
            << "#define WIDTH " << param.width << '\n'
            << "#define HEIGHT " << param.height << '\n'
            << "#define STRIDE " << get_stride(param.width) << '\n'
            << "#define SIGMA_SPATIAL_SCALED " << std::fixed << param.sigma_spatial_scaled << "f\n"
            << "#define SIGMA_COLOR_SCALED " << std::fixed << param.sigma_color_scaled << "f\n"
            << "#define RADIUS " << param.radius << '\n'
            << "#define USE_SHARED_MEMORY " << (int) param.use_shared_memory << '\n'
            << kernel_code_raw;

        std::string_view kernel_code = program_stream.view();
        const char * kernel_code_ptrs[] { kernel_code.data() };

        size_t size = std::size(kernel_code) - 1;
        auto program = CALL(clCreateProgramWithSource, context, 1, kernel_code_ptrs, &size);

        const char * options[] {
            "-cl-denorms-are-zero",
            "-cl-fast-relaxed-math"
        };
        if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
            size_t size;
            checkError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size));
            std::string log;
            log.resize(size);
            checkError(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr));
            std::cerr << log.c_str() << std::endl;
            abort();
        }

        return program;
    }

    static inline cl_kernel init_kernel(
        cl_program program,
        const char * kernel_name
    ) noexcept {
        return CALL(clCreateKernel, program, kernel_name);
    }

    static inline std::vector<cl_command_queue> init_command_queue(
        cl_context context,
        cl_device_id device,
        int num_resources
    ) noexcept {
        std::vector<cl_command_queue> command_queues;
        command_queues.reserve(num_resources);
        cl_command_queue_properties properties {};
        for (int i = 0; i < num_resources; ++i) {
            command_queues.emplace_back(CALL(clCreateCommandQueue, context, device, properties));
        }
        return command_queues;
    }

public:
    inline void execute(
        float * dstp,
        const float * srcp,
        int plane,
        int stride
    ) noexcept {

        this->semaphore.acquire();

        int ticket;

        {
            while (this->tickets_lock.test_and_set(std::memory_order_acquire)) {
                // spin
            }

            ticket = this->tickets.back();
            this->tickets.pop_back();

            this->tickets_lock.clear(std::memory_order_release);
        }

        this->resources[ticket].execute(
            dstp, this->kernels[ticket][plane], srcp, stride,
            plane, this->command_queues[ticket]);

        {
            while (this->tickets_lock.test_and_set(std::memory_order_acquire)) {
                // spin
            }

            this->tickets.push_back(ticket);

            this->tickets_lock.clear(std::memory_order_release);
        }

        this->semaphore.release();
    }

    ~ResourcePool() noexcept {
        if (this->context != nullptr) {
            for (auto & resource : resources) {
                resource.destroy();
            }
            for (auto command_queue : this->command_queues) {
                checkError(clReleaseCommandQueue(command_queue));
            }
            for (auto kernels : this->kernels) {
                for (auto kernel : kernels) {
                    if (kernel) {
                        checkError(clReleaseKernel(kernel));
                    }
                }
            }
            for (auto program : this->programs) {
                if (program) {
                    checkError(clReleaseProgram(program));
                }
            }
            checkError(clReleaseContext(this->context));
        }
    }
};


struct BilateralData {
    VSNodeRef * node;
    const VSVideoInfo * vi;

    std::array<bool, 3> process { true, true, true };

    ResourcePool resource_pool;
};


static void VS_CC BilateralInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) {

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

        auto set_error = [&](const std::string & error_message) {
            vsapi->setFilterError(("BilateralGPU: " + error_message).c_str(), frameCtx);
            vsapi->freeFrame(src);
            return nullptr;
        };

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!d->process[plane]) {
                continue;
            }

            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            int s_pitch = vsapi->getStride(src, plane);
            int bps = d->vi->format->bitsPerSample;
            int s_stride = s_pitch / d->vi->format->bytesPerSample;
            auto srcp = vsapi->getReadPtr(src, plane);
            auto dstp = vsapi->getWritePtr(dst, plane);

            if (bps == 32) {
                d->resource_pool.execute(
                    (float *) dstp,
                    (const float *) srcp,
                    plane,
                    s_stride
                );
            } else {
                abort();
            }
        }

        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}


static void VS_CC BilateralFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) {

    BilateralData * d = static_cast<BilateralData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}


static void VS_CC BilateralCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {

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
        sample == stInteger ||
        (sample == stFloat && bps != 32)
    ) {

        return set_error("only constant format 32bit float input supported");
    }

    int error;

    std::array<float, 3> sigma_spatial;
    for (unsigned i = 0; i < std::size(sigma_spatial); ++i) {
        sigma_spatial[i] = static_cast<float>(
            vsapi->propGetFloat(in, "sigma_spatial", i, &error));

        if (error) {
            if (i == 0) {
                sigma_spatial[i] = 3.0f;
            } else if (i == 1) {
                auto subH = d->vi->format->subSamplingH;
                auto subW = d->vi->format->subSamplingW;
                sigma_spatial[i] = static_cast<float>(
                    sigma_spatial[i] / std::sqrt((1 << subH) * (1 << subW)));
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

    std::array<float, 3> sigma_color;
    for (unsigned i = 0; i < std::size(sigma_color); ++i) {
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

    std::array<Params, 3> params;

    for (auto & param : params) {
        param.block_x = 16;
        param.block_y = 8;
    }

    for (unsigned i = 0; i < std::size(params); ++i) {
        if (i == 0) {
            params[i].width = d->vi->width;
            params[i].height = d->vi->height;
        } else {
            params[i].width = d->vi->width >> d->vi->format->subSamplingW;
            params[i].height = d->vi->height >> d->vi->format->subSamplingH;
        }
    }

    for (unsigned i = 0; i < std::size(params); ++i) {
        if (sigma_color[i] < FLT_EPSILON) {
            d->process[i] = false;
        } else {
            float value { -0.5f * std::numbers::log2e_v<float> / (sigma_color[i] * sigma_color[i]) };
            params[i].sigma_color_scaled = value;
        }
    }

    for (unsigned i = 0; i < std::size(params); ++i) {
        params[i].radius = int64ToIntS(vsapi->propGetInt(in, "radius", i, &error));

        if (error) {
            params[i].radius = std::max(1, static_cast<int>(std::roundf(sigma_spatial[i] * 3.f)));
        } else if (params[i].radius <= 0) {
            return set_error("\"radius\" must be positive");
        }
    }

    for (unsigned i = 0; i < std::size(params); ++i) {
        float value { -0.5f * std::numbers::log2e_v<float> / (sigma_spatial[i] * sigma_spatial[i]) };
        params[i].sigma_spatial_scaled = value;
    }

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    auto num_streams = int64ToIntS(vsapi->propGetInt(in, "num_streams", 0, &error));
    if (error) {
        num_streams = 1;
    }

    bool use_shared_memory = !!vsapi->propGetInt(in, "use_shared_memory", 0, &error);
    if (error) {
        use_shared_memory = true;
    }

    for (auto & param : params) {
        param.use_shared_memory = use_shared_memory;
    }

    d->resource_pool.init(
        device_id,
        params,
        num_streams,
        d->process
    );

    vsapi->createFilter(
        in, out, "Bilateral",
        BilateralInit, BilateralGetFrame, BilateralFree,
        fmParallel, 0, d.release(), core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) {

    configFunc(
        "com.wolframrhodium.bilateralgpu_cl", "bilateralgpu_cl", "Bilateral filter using OpenCL",
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

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
