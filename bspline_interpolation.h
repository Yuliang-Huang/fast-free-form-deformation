#pragma once

#include "control_point_grid.h"
#include <cuda_runtime.h>

namespace BSplineInterpolation 
{
    // Forward pass: Interpolate displacement and jacobian
    void forward(const float3* inputPoints, ControlPointGrid& grid, float* outputDisplacements, float3* outputJacobian, int numPoints);

    // overloaded forward pass
    void forward(const float3* inputPoints, ControlPointGrid& grid, float* outputs, int numPoints);

    // Backward pass: Compute gradients to input points and control points
    void backward(const float3* inputPoints, ControlPointGrid& grid, float3* gradControlPoints, const float3* gradDisplacements, const float* gradJacobians, int numPoints);

    // overloaded backward pass
    void backward(const float3* inputPoints, ControlPointGrid& grid, float* gradControlPoints, const float* gradOutputs, int numPoints);
};
