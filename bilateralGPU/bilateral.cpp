//Copyright (c) 2017 WolframRhodium. All rights reserved.

#include <opencv2\cudaimgproc.hpp>
#include <vapoursynth\VapourSynth.h>
#include <vapoursynth\VSHelper.h>

struct BilateralData
{
	VSNodeRef * node;
	VSVideoInfo vi;
	float sigma_spatial[3], sigma_color[3];
	int kernel_size[3], borderMode[3], deviceId[3];
	bool process[3];
};

template < typename T >
static void process(const VSFrameRef * src, VSFrameRef * dst, BilateralData * d, const VSAPI * vsapi)
{
	auto sampleType = d->vi.format->sampleType == stFloat ? CV_32FC1 : d->vi.format->bitsPerSample == 8 ? CV_8UC1 : CV_16UC1;

	for (int plane = 0; plane < d->vi.format->numPlanes; plane++)
		if (d->process[plane]) {
			if (d->deviceId[plane] != 0) {
				cv::cuda::setDevice(d->deviceId[plane]);
			}

			const unsigned width = vsapi->getFrameWidth(src, plane);
			const unsigned height = vsapi->getFrameHeight(src, plane);

			const unsigned srcStride = vsapi->getStride(src, plane) / sizeof(T);
			const unsigned dstStride = vsapi->getStride(dst, plane) / sizeof(T);

			const T *srcp = reinterpret_cast<const T *>(vsapi->getReadPtr(src, plane));
			T* VS_RESTRICT dstp = reinterpret_cast<T *>(vsapi->getWritePtr(dst, plane));

			cv::Mat srcImg(cv::Size(width, height), sampleType);
			cv::Mat dstImg(cv::Size(width, height), sampleType);

			for (unsigned y = 0; y < height; y++) {
				for (unsigned x = 0; x < width; x++) {
					srcImg.at<T>(y, x) = srcp[x];
				}

				srcp += srcStride;
			}

			// device memory
			cv::cuda::GpuMat input;
			cv::cuda::GpuMat output;

			// allocate & copy data from host to device
			input.upload(srcImg);

			cv::cuda::bilateralFilter(input, output, d->kernel_size[plane], d->sigma_color[plane], d->sigma_spatial[plane], d->borderMode[plane]);

			// copy data from device to host
			output.download(dstImg);

			for (unsigned y = 0; y < height; y++) {
				for (unsigned x = 0; x < width; x++) {
					dstp[x] = dstImg.at<T>(y, x);
				}

				dstp += dstStride;
			}

			input.release();
			output.release();
		}
}

static void VS_CC BilateralInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
	BilateralData * d = static_cast<BilateralData *>(*instanceData);
	vsapi->setVideoInfo(&d->vi, 1, node);
}

static const VSFrameRef *VS_CC BilateralGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
	BilateralData * d = static_cast<BilateralData *>(*instanceData);

	if (activationReason == arInitial) {
		vsapi->requestFrameFilter(n, d->node, frameCtx);
	} else if (activationReason == arAllFramesReady) {
		const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
		const int planes[] = { 0, 1, 2 };
		const VSFrameRef * cp_planes[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
		VSFrameRef * dst = vsapi->newVideoFrame2(d->vi.format, d->vi.width, d->vi.height, cp_planes, planes, src, core);

		if (d->vi.format->sampleType == stInteger) {
			if (d->vi.format->bytesPerSample == 1)
			{
				process<uint8_t>(src, dst, d, vsapi);
			}
			else if (d->vi.format->bytesPerSample == 2)
			{
				process<uint16_t>(src, dst, d, vsapi);
			}
		} else {
			process<float>(src, dst, d, vsapi);
		}

		vsapi->freeFrame(src);

		return dst;
	}

	return nullptr;
}

static void VS_CC BilateralFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
	BilateralData * d = static_cast<BilateralData *>(instanceData);

	vsapi->freeNode(d->node);

	delete d;
}

static void VS_CC BilateralCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
	BilateralData d{};
	int i, m, n, o;

	d.node = vsapi->propGetNode(in, "clip", 0, nullptr);
	d.vi = *vsapi->getVideoInfo(d.node);

	if (!isConstantFormat(&d.vi) || (d.vi.format->sampleType == stFloat && d.vi.format->bitsPerSample != 32)) {
		vsapi->setError(out, "Bilateral: only constant format 8-16 integer or 32-bit float input supported.");
		vsapi->freeNode(d.node);
		return;
	}

	m = vsapi->propNumElements(in, "sigma_spatial");
	for (i = 0; i < 3; i++)
	{
		if (i < m)
		{
			d.sigma_spatial[i] = (float) vsapi->propGetFloat(in, "sigma_spatial", i, nullptr);
		}
		else if (i == 0)
		{
			d.sigma_spatial[0] = 1.0f;
		}
		else if (i == 1 && (d.vi.format->colorFamily == cmYUV || d.vi.format->colorFamily == cmYCoCg) && d.vi.format->subSamplingH && d.vi.format->subSamplingW) // Reduce sigmaS for sub-sampled chroma planes by default
		{
			d.sigma_spatial[1] = (float) (d.sigma_spatial[0] / std::sqrt((1 << d.vi.format->subSamplingH)*(1 << d.vi.format->subSamplingW)));
		}
		else
		{
			d.sigma_spatial[i] = d.sigma_spatial[i - 1];
		}

		if (d.sigma_spatial[i] < 0)
		{
			vsapi->setError(out, "Bilateral: \"sigma_spatial\" must be greater than zero.");
			return;
		}
	}

	m = vsapi->propNumElements(in, "sigma_color");
	for (i = 0; i < 3; i++)
	{
		if (i < m)
		{
			d.sigma_color[i] = (float) vsapi->propGetFloat(in, "sigma_color", i, nullptr);
		}
		else if (i == 0)
		{
			d.sigma_color[i] = 1.0f;
		}
		else
		{
			d.sigma_color[i] = d.sigma_color[i - 1];
		}

		if (d.sigma_color[i] < 0)
		{
			vsapi->setError(out, "Bilateral: \"sigma_color\" must be greater than zero.");
			return;
		}
	}

	n = d.vi.format->numPlanes;
	m = vsapi->propNumElements(in, "planes");
	for (i = 0; i < 3; i++)
	{
		if (i > 0 && (d.vi.format->colorFamily == cmYUV || d.vi.format->colorFamily == cmYCoCg)) // Chroma planes are not processed by default
			d.process[i] = false;
		else
			d.process[i] = m <= 0;
	}
	for (i = 0; i < m; i++) {
		o = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));
		if (o < 0 || o >= n)
		{
			vsapi->setError(out, "Bilateral: plane index out of range");
			return;
		}
		if (d.process[o])
		{
			vsapi->setError(out, "Bilateral: plane specified twice");
			return;
		}
		d.process[o] = true;
	}
	for (i = 0; i < 3; i++)
	{
		if (d.sigma_spatial[i] == 0 || d.sigma_color[i] == 0)
			d.process[i] = 0;
	}

	m = vsapi->propNumElements(in, "kernel_size");
	for (i = 0; i < 3; i++)
	{
		if (i < m)
		{
			d.kernel_size[i] = int64ToIntS(vsapi->propGetInt(in, "kernel_size", i, nullptr));
		}
		else if (i == 0)
		{
			d.kernel_size[i] = 0; // sigma_spatial * 3
		}
		else
		{
			d.kernel_size[i] = d.kernel_size[i - 1];
		}

		if (d.kernel_size[i] < 0)
		{
			vsapi->setError(out, "Bilateral: \"kernel_size\" can not be less than zero.");
			return;
		}
	}

	m = vsapi->propNumElements(in, "borderMode");
	for (i = 0; i < 3; i++)
	{
		if (i < m)
		{
			d.borderMode[i] = int64ToIntS(vsapi->propGetInt(in, "borderMode", i, nullptr));
		}
		else if (i == 0)
		{
			d.borderMode[i] = 4; // cv::BORDER_DEFAULT
		}
		else
		{
			d.borderMode[i] = d.borderMode[i - 1];
		}

		if (d.borderMode[i] < 1 || d.borderMode[i] > 4)
		{
			vsapi->setError(out, "Bilateral: \"borderMode\" can not be less than zero.");
			return;
		}
	}

	m = vsapi->propNumElements(in, "device");
	for (i = 0; i < 3; i++)
	{
		if (i < m)
		{
			d.deviceId[i] = int64ToIntS(vsapi->propGetInt(in, "device", i, nullptr));
		}
		else if (i == 0)
		{
			d.deviceId[i] = 0;
		}
		else
		{
			d.deviceId[i] = d.deviceId[i - 1];
		}
	}

	BilateralData * data = new BilateralData{ d };

	vsapi->createFilter(in, out, "Bilateral", BilateralInit, BilateralGetFrame, BilateralFree, fmParallelRequests, 0, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
	configFunc("com.wolframrhodium.bilateralGPU", "bilateralgpu", "Bilateral filter using CUDA", VAPOURSYNTH_API_VERSION, 1, plugin);
	registerFunc("Bilateral", "clip:clip;sigma_spatial:float[]:opt;sigma_color:float[]:opt;planes:int[]:opt;kernel_size:int[]:opt;borderMode:int[]:opt;device:int[]:opt",
		BilateralCreate, nullptr, plugin);
}
