#pragma once

#include "control_point_grid.h"
#include <cuda_runtime.h>

// Forward pass kernel
__global__ void forwardKernel(
    cudaTextureObject_t * texArray, 
    const float3* inputPoints, 
    float* outputDisplacements, 
    float3* outputJacobian, 
    int numPoints,
    int width, 
    int height, 
    int depth
);

//overloaded forward pass kernel
__global__ void forwardKernel(
    cudaTextureObject_t * texArray, 
    const float3* inputPoints, 
    float* outputs, 
    int numPoints,
    int width, 
    int height, 
    int depth,
    int nchannels
);
