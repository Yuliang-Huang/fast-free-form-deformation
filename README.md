# Fast Free-Form Deformation (FFD)

This repository provides a fast and efficient implementation of Free-Form Deformation (FFD), developed as part of the MICCAI 2025 paper:  
**"DIGS: Dynamic CBCT Reconstruction using Deformation-Informed 4D Gaussian Splatting and a Low-Rank Free-Form Deformation Model."**

## Features
- Custom CUDA kernels compatible with PyTorch, enabling seamless integration into any learning-based pipeline.
- Supports cubic B-spline interpolation at arbitrary points in 3D space from an n-channel 3D control point grid of shape `[C, Nx, Ny, Nz]`, where `C` is the number of channels.
- The grid size `(Nx, Ny, Nz)` is automatically computed based on the number of voxels (`nvox`) and grid spacing (`grid_spacing_in_voxels`).
- Supports differentiable Jacobian computation when the channel dimension `C = 3` (deformation fields).
- Generalizes to arbitrary feature channels in 3D space without Jacobian computation, useful for interpolating Gaussian attributes as in the referenced paper.

## Installation
```bash
git clone https://github.com/Yuliang-Huang/fast-free-form-deformation.git
cd fast-free-form-deformation
python setup.py install
```

## Usage

```python
from ffd import FFD

# Create model
model = FFD(nvox, dvox, grid_spacing_in_voxels)

# Get 3D displacement vector field and Jacobian
displacement, jacobian = model(input_points, control_point_grid)

# For n-dimensional feature interpolation without Jacobian
model = FFD(nvox, dvox, grid_spacing_in_voxels, nchannels=n, return_jacobian=False)
features = model(input_points, control_point_grid)
```

## Notes

- The **input points** must be provided in **real-world coordinates**, where the origin is assumed to be at the center of the volume defined by:
  - `nvox`: number of voxels along each axis (tuple of 3 ints), and
  - `dvox`: voxel spacing along each axis (tuple of 3 floats).

- The **control point grid**:
  - Should have shape `[C, Nx, Ny, Nz]` where `C = 3` for displacement fields or `C = n` for n-dimensional feature interpolation.
  - The grid size `(Nx, Ny, Nz)` is automatically computed based on `nvox` and `grid_spacing_in_voxels`, and is accessible via `model.grid_dim`.

- The model supports **arbitrary feature channels** in 3D space, though **Jacobian computation** is only supported when `C = 3`, i.e., for deformation fields.

## Citation

Please cite our paper if you find this repository useful for your research
```bib
@misc{huang2025digs,
      title={DIGS: Dynamic CBCT Reconstruction using Deformation-Informed 4D Gaussian Splatting and a Low-Rank Free-Form Deformation Model}, 
      author={Yuliang Huang and Imraj Singh and Thomas Joyce and Kris Thielemans and Jamie R. McClelland},
      year={2025},
      eprint={2506.22280},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.22280}, 
}
``` 

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
