import torch
import torch.nn as nn
import numpy as np
from ._C import ControlPointGrid
from ._C import forwardBsplineInterpolationWithJacobian, forwardBsplineInterpolationWithoutJacobian
from ._C import backwardBsplineInterpolationWithJacobian, backwardBsplineInterpolationWithoutJacobian

class _GetDisplacementAndJacobian(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        grid: torch.Tensor,
        grid_container: ControlPointGrid,
    ):
        grid_container.copyData(grid.permute(3,2,1,0).contiguous())
        displacement, jacobian = forwardBsplineInterpolationWithJacobian(grid_container, points)
        jacobian += torch.eye(3).cuda().view(1, 3, 3).expand_as(jacobian)
        ctx.save_for_backward(points)
        ctx.grid_container = grid_container
        return displacement, jacobian

    @staticmethod
    def backward(
        ctx, 
        grad_displacement, 
        grad_jacobian
    ):
        if grad_jacobian.isnan().any():
            print("grad_jacobian is nan")
        if grad_displacement.isnan().any():
            print("grad_displacement is nan")
        grid_container = ctx.grid_container
        points = ctx.saved_tensors[0]
        if points.isnan().any():
            print("points is nan")
        grad_grid = backwardBsplineInterpolationWithJacobian(grid_container, points, grad_displacement.contiguous(), grad_jacobian.contiguous())
        if grad_grid.isnan().any():
            print("grad_grid is nan")
        return None, grad_grid, None

class _GetDelta(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        grid: torch.Tensor,
        grid_container: ControlPointGrid,
    ):
        grid_container.copyData(grid.permute(3,2,1,0).contiguous())
        delta = forwardBsplineInterpolationWithoutJacobian(grid_container, points)
        ctx.save_for_backward(points)
        ctx.grid_container = grid_container
        return delta

    @staticmethod
    def backward(
        ctx, 
        grad_delta, 
    ):
        grid_container = ctx.grid_container
        points = ctx.saved_tensors[0]
        grad_grid = backwardBsplineInterpolationWithoutJacobian(grid_container, points, grad_delta)
        return None, grad_grid, None


class FFD(nn.Module):

    def __init__(self,
                 nvox,
                 dvox,
                 grid_spacing_in_voxels,
                 nchannels=3,
                 return_jacobian=True):
        super(FFD, self).__init__()
        grid_dim = [int(nvox[i]/grid_spacing_in_voxels[i])+3 for i in range(3)]
        self._grid = ControlPointGrid(grid_dim[0], grid_dim[1], grid_dim[2], nchannels)
        self.grid_dim = grid_dim
        if return_jacobian:
            assert nchannels == 3, "Jacobian is only supported for 3 channels."
        self.return_jacobian = return_jacobian
        self.nchannels = nchannels
        self.grid_spacing_in_scene_units = torch.tensor([grid_spacing_in_voxels[i]*dvox[i] for i in range(3)],dtype=torch.float32).cuda()
        self.grid_origin = torch.tensor([-dvox[i]*nvox[i]/2 for i in range(3)],dtype=torch.float32).cuda() - self.grid_spacing_in_scene_units

    def forward(self, points, grid):
        points_in_voxels = (points - self.grid_origin) /self.grid_spacing_in_scene_units
        if self.return_jacobian:
            displacement, jacobian = _GetDisplacementAndJacobian.apply(points_in_voxels, grid, self._grid)
            return displacement, jacobian
        else:
            delta = _GetDelta.apply(points_in_voxels, grid, self._grid)
            return delta
