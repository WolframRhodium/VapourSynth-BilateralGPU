# VapourSynth-BilateralGPU
Copyright© 2021 WolframRhodium

Bilateral filter in CUDA for VapourSynth.

## Description
[Bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter) is a non-linear, edge-preserving and noise-reducing smoothing filter for images.

The intensity value at each pixel in an image is replaced by a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution.

Special thanks to [Kice](https://github.com/kice) for doing most of the work in previous implementation. Part of the support code is copied from [CPU-based Bilateral filter for VapourSynth](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral).

## Requirements
CUDA-Enabled GPU(s)

## Supported Formats

sample type: 8-16 bit integer or 32 bit float Gray/YUV/RGB/YCoCg input

## Usage

```python
core.bilateralgpu.Bilateral(clip clip, float[] sigma_spatial=3.0, float[] sigma_color=0.02, int[] radius=0, int[] device_id=0)
```

- clip:
    The input clip.

- sigma_spatial: (Default: 3.0)
    Filter sigma in the coordinate space.
	Use an array to assign it for each plane. If "sigma_spatial" for the second plane is not specified, it will be set according to the sigma_spatial of first plane and sub-sampling.

- sigma_color: (Default: 0.02)
    Filter sigma in the color space.
	Use an array to assign it for each plane, otherwise the same sigma_color is used for all the planes.
	It will be normalized internally, so that for clips with different bit depths, the same values get similar results.

- radius: (Default: 0)
    Kernel window size. 0 = auto calculate based on "sigma_spatial".

- device_id: (Default: 0)
    CUDA device ID.
