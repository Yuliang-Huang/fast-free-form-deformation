from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='ffd',
    packages=['ffd'],
    ext_modules=[
        CUDAExtension(
            name='ffd._C', 
            sources=[
                'backward.cu',
                'forward.cu',
                'bspline_interpolation.cu',
                'control_point_grid.cu',
                'pytorch_interface.cu',
                'ext.cpp'
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
