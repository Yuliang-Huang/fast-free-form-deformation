#include "bspline_interpolation.h"
#include "forward.h"
#include "backward.h"

void BSplineInterpolation::forward(
    const float3* inputPoints, 
    ControlPointGrid& grid, 
    float* outputDisplacements, 
    float3* outputJacobian, 
    int numPoints
) 
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numPoints + 128 - 1) / 128, 3, 1);

    forwardKernel<<<gridSize, blockSize>>>(
        grid.texArray,
        inputPoints, 
        outputDisplacements, 
        outputJacobian, 
        numPoints,
        grid.width, 
        grid.height, 
        grid.depth
    );
    auto ret = cudaDeviceSynchronize(); 
    if (ret != cudaSuccess) { 
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); 
        throw std::runtime_error(cudaGetErrorString(ret)); 
    } 

}

//overloaded forward function
void BSplineInterpolation::forward(
    const float3* inputPoints, 
    ControlPointGrid& grid, 
    float* outputs, 
    int numPoints
) 
{
    dim3 blockSize(128, 1, 1);
    dim3 gridSize((numPoints + 128 - 1) / 128, grid.nchannels, 1);

    forwardKernel<<<gridSize, blockSize>>>(
        grid.texArray,
        inputPoints, 
        outputs, 
        numPoints,
        grid.width, 
        grid.height, 
        grid.depth,
        grid.nchannels
    );
    auto ret = cudaDeviceSynchronize(); 
    if (ret != cudaSuccess) { 
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); 
        throw std::runtime_error(cudaGetErrorString(ret)); 
    } 

}

void BSplineInterpolation::backward(
    const float3* inputPoints, 
    ControlPointGrid& grid, 
    float3* gradControlPoints, 
    const float3* gradDisplacements, 
    const float* gradJacobians, 
    int numPoints
) 
{
    int blockSize = 128;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    backwardKernel<<<gridSize, blockSize>>>(
        grid.texArray[0], 
        grid.texArray[1], 
        grid.texArray[2],
        inputPoints, 
        gradControlPoints,
        gradDisplacements, 
        gradJacobians, 
        numPoints,
        grid.width, 
        grid.height, 
        grid.depth
    );

    auto ret = cudaDeviceSynchronize(); 
    if (ret != cudaSuccess) { 
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); 
        throw std::runtime_error(cudaGetErrorString(ret)); 
    } 

}

// overloaded backward function
void BSplineInterpolation::backward(
    const float3* inputPoints, 
    ControlPointGrid& grid, 
    float* gradControlPoints, 
    const float* gradOutputs, 
    int numPoints
) 
{
    int blockSize = 128;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    backwardKernel<<<gridSize, blockSize>>>(
        grid.texArray, 
        inputPoints,  
        gradControlPoints,
        gradOutputs, 
        numPoints,
        grid.width, 
        grid.height, 
        grid.depth,
        grid.nchannels
    );

    auto ret = cudaDeviceSynchronize(); 
    if (ret != cudaSuccess) { 
        std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); 
        throw std::runtime_error(cudaGetErrorString(ret)); 
    } 

}