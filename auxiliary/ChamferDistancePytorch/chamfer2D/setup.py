from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_2D',
    ext_modules=[
        CUDAExtension('chamfer_2D', [
            'chamfer_cuda.cpp',
            'chamfer2D.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
