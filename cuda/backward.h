#pragma once

#include "control_point_grid.h"
#include <cuda_runtime.h>

__global__ void backwardKernel(
    cudaTextureObject_t texX, 
    cudaTextureObject_t texY, 
    cudaTextureObject_t texZ,
    const float3* inputPoints, 
    float3* gradControlPoints,
    const float3* gradDisplacements, 
    const float* gradJacobians, 
    int numPoints,
    int width, 
    int height, 
    int depth
);

//overloaded backward pass kernel
__global__ void backwardKernel(
    cudaTextureObject_t * texArray, 
    const float3* inputPoints,
    float* gradControlPoints,
    const float* gradOutputs,
    int numPoints,
    int width, 
    int height, 
    int depth,
    int nchannels
);