# VapourSynth-BilateralGPU
Copyright© 2017 WolframRhodium

Bilateral filter for VapourSynth based on the OpenCV-CUDA library.
## Description
[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images.

The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

Special thanks to [Kice](https://github.com/kice) for doing most of the work. Part of the support code is copied from [CPU-based Bilateral filter for VapourSynth](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral).

## Requirements
CUDA-Enabled GPU(s), OpenCV run-time library with CUDA module

## Supported Formats

sample type: 8-16 bit integer or 32 bit float Gray/YUV/RGB/YCoCg input

## Usage

```python
core.bilateralgpu.Bilateral(clip clip, float[] sigma_spatial=1.0, float[] sigma_color=1.0, int[] planes, int[] kernel_size=0, int[] borderMode=4, int[] device=0)
```

- clip:
    The input clip.

- sigma_spatial: (Default: 1.0)
    Filter sigma in the coordinate space.
	Use an array to assign it for each plane. If sigma_spatial for the second plane is not specified, it will be set according to the sigma_spatial of first plane and sub-sampling.

- sigma_color: (Default: 1.0)
    Filter sigma in the color space.
	Use an array to assign it for each plane, otherwise the same sigma_color is used for all the planes.
	It will be normalized internally, so that for clips with different bit depths, the same values get similar results.

- planes:
    An array to specify which planes to process.
    By default, chroma planes are not processed.

- kernel_size: (Default: 0)
    Kernel window size. If set to 0, it will be automatically calculated by the filter according to the "sigma_spatial".

- borderMode: (Default: 4)
    Border type. 1: cv::BORDER_REPLICATE, 2: cv::BORDER_REFLECT, 3: cv::BORDER_WRAP, 4: cv::BORDER_REFLECT_101.
	See [cv::BorderTypes](http://docs.opencv.org/3.2.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5) for details.

- device: (Default: 0)
    CUDA device ID.
