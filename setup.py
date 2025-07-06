from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name='ffd',
    packages=['ffd'],
    ext_modules=[
        CUDAExtension(
            name='ffd._C', 
            sources=[
                'cuda/backward.cu',
                'cuda/forward.cu',
                'cuda/bspline_interpolation.cu',
                'cuda/control_point_grid.cu',
                'cuda/pytorch_interface.cu',
                'cuda/ext.cpp'
            ],
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)),'cuda')]
                }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
