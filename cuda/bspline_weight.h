#pragma once
#include <cuda_runtime.h>

__forceinline__ __device__ void computeBSplineWeights(float t, float B[4])
{
    float t2 = t * t;
    float t3 = t2 * t;

    B[0] = (-t3 + 3.0f * t2 - 3.0f * t + 1.0f) / 6.0f;
    B[1] = (3.0f * t3 - 6.0f * t2 + 4.0f) / 6.0f;
    B[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
    B[3] = t3 / 6.0f;
}

__forceinline__ __device__ void computeBSplineWeightsDerivative(float t, float dB[4])
{
    float t2 = t * t;

    dB[0] = (-3.0f * t2 + 6.0f * t - 3.0f) / 6.0f;
    dB[1] = (9.0f * t2 - 12.0f * t) / 6.0f;
    dB[2] = (-9.0f * t2 + 6.0f * t + 3.0f) / 6.0f;
    dB[3] = (3.0f * t2) / 6.0f;
}

__forceinline__ __device__ void computeBSplineWeightsSecondDerivative(float t, float d2B[4])
{
    d2B[0] = (-6.0f * t + 6.0f) / 6.0f;
    d2B[1] = (18.0f * t - 12.0f) / 6.0f;
    d2B[2] = (-18.0f * t + 6.0f) / 6.0f;
    d2B[3] = (6.0f * t) / 6.0f;
}