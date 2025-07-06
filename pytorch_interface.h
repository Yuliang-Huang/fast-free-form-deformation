#pragma once
#include <torch/extension.h>
#include <tuple>
#include "control_point_grid.h"

std::tuple<torch::Tensor, torch::Tensor> forwardBsplineInterpolationWithJacobian
(
	ControlPointGrid& grid,
    torch::Tensor& points
);

torch::Tensor forwardBsplineInterpolationWithoutJacobian
(
	ControlPointGrid& grid,
    torch::Tensor& points
);

torch::Tensor backwardBsplineInterpolationWithJacobian
(
	ControlPointGrid& grid,
    torch::Tensor& points,
    torch::Tensor& gradDisplacements,
    torch::Tensor& gradJacobian
);

torch::Tensor backwardBsplineInterpolationWithoutJacobian
(
    ControlPointGrid& grid,
    torch::Tensor& points,
    torch::Tensor& gradOutputs
);