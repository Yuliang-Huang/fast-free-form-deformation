#include "pytorch_interface.h"
#include "bspline_interpolation.h"
#include <cuda_runtime.h>

std::tuple<torch::Tensor, torch::Tensor> forwardBsplineInterpolationWithJacobian
(
	ControlPointGrid& grid,
    torch::Tensor& points
)
{
    // Get the number of points
    int numPoints = points.size(0);
    auto floatOptions = points.options().dtype(torch::kFloat32);

    // Create output tensors
    torch::Tensor displacements = torch::zeros({numPoints, 3}, floatOptions).contiguous();
    torch::Tensor jacobians = torch::zeros({numPoints, 3, 3}, floatOptions).contiguous();

    // Call the forward kernel
    BSplineInterpolation::forward(
        (float3*)points.data_ptr<float>(),
        grid,
        displacements.data_ptr<float>(),
        (float3*)jacobians.data_ptr<float>(),
        numPoints
    );

    return std::make_tuple(displacements, jacobians);
}

torch::Tensor forwardBsplineInterpolationWithoutJacobian
(
	ControlPointGrid& grid,
    torch::Tensor& points
)
{
    // Get the number of points
    int numPoints = points.size(0);
    auto floatOptions = points.options().dtype(torch::kFloat32);

    // Create output tensors
    torch::Tensor output = torch::zeros({numPoints, grid.nchannels}, floatOptions).contiguous();

    // Call the forward kernel
    BSplineInterpolation::forward(
        (float3*)points.data_ptr<float>(),
        grid,
        output.data_ptr<float>(),
        numPoints
    );

    return output;
}

torch::Tensor backwardBsplineInterpolationWithJacobian
(
	ControlPointGrid& grid,
    torch::Tensor& points,
    torch::Tensor& gradDisplacements,
    torch::Tensor& gradJacobian
)
{
    // Get the number of points
    int numPoints = points.size(0);
    auto floatOptions = points.options().dtype(torch::kFloat32);

    // Create output tensors
    torch::Tensor gradControlPoints = torch::zeros({grid.width, grid.height, grid.depth, 3}, floatOptions).contiguous();

    // Call the backward kernel
    BSplineInterpolation::backward(
        (float3*)points.data_ptr<float>(),
        grid,
        (float3*)gradControlPoints.data_ptr<float>(),
        (float3*)gradDisplacements.data_ptr<float>(),
        (float*)gradJacobian.data_ptr<float>(),
        numPoints
    );

    return gradControlPoints;
}

torch::Tensor backwardBsplineInterpolationWithoutJacobian
(
    ControlPointGrid& grid,
    torch::Tensor& points,
    torch::Tensor& gradOutputs
)
{
    // Get the number of points
    int numPoints = points.size(0);
    auto floatOptions = points.options().dtype(torch::kFloat32);

    // Create output tensors
    torch::Tensor gradControlPoints = torch::zeros({grid.width, grid.height, grid.depth, grid.nchannels}, floatOptions).contiguous();

    // Call the backward kernel
    BSplineInterpolation::backward(
        (float3*)points.data_ptr<float>(),
        grid,
        gradControlPoints.data_ptr<float>(),
        gradOutputs.data_ptr<float>(),
        numPoints
    );

    return gradControlPoints;
}