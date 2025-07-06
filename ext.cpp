#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "control_point_grid.h"
#include "pytorch_interface.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  
  m.def("forwardBsplineInterpolationWithJacobian", &forwardBsplineInterpolationWithJacobian, "Forward Bspline Interpolation with Jacobian");
  m.def("forwardBsplineInterpolationWithoutJacobian", &forwardBsplineInterpolationWithoutJacobian, "Forward Bspline Interpolation without Jacobian");
  m.def("backwardBsplineInterpolationWithJacobian", &backwardBsplineInterpolationWithJacobian, "Backward Bspline Interpolation with Jacobian");
  m.def("backwardBsplineInterpolationWithoutJacobian", &backwardBsplineInterpolationWithoutJacobian, "Backward Bspline Interpolation without Jacobian");
  
  py::class_<ControlPointGrid>(m, "ControlPointGrid")
    .def(py::init<int, int, int, int>())
    .def("copyData", &ControlPointGrid::copyData);
}