from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='patchworktorch',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('patchworktorch_backend', 
            sources=['src/patchworkpp.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']})
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
    ]
)