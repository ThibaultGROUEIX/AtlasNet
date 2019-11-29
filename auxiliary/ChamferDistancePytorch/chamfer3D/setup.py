from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            'chamfer_cuda.cpp',
            'chamfer3D.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
