from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_5D',
    ext_modules=[
        CUDAExtension('chamfer_5D', [
            'chamfer_cuda.cpp',
            'chamfer5D.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
