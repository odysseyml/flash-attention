import os
import subprocess
from packaging.version import parse, Version

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        nvcc_threads = os.getenv("NVCC_THREADS") or "4"
        return nvcc_extra_args + ["--threads", nvcc_threads]
    return nvcc_extra_args


# Odyssey patch: explicit gencode list matching flash-attn root setup.py defaults
# ("80;90;100;120"). Upstream fused_dense_lib setup.py had no -gencode flags at all,
# so torch must auto-detect archs from a local GPU -- which fails on CPU-only CI
# runners ("IndexError" in _get_cuda_arch_flags). Pinning archs here also targets
# Ampere + Hopper + Blackwell B200 (sm_100) + RTX PRO 6000 / RTX 50-series (sm_120).
cc_flag = []
_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
if bare_metal_version >= Version("11.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90,code=sm_90")
if bare_metal_version >= Version("12.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_100,code=sm_100")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")


setup(
    name='fused_dense_lib',
    ext_modules=[
        CUDAExtension(
            name='fused_dense_lib',
            sources=['fused_dense.cpp', 'fused_dense_cuda.cu'],
            extra_compile_args={
                               'cxx': ['-O3',],
                               'nvcc': append_nvcc_threads(['-O3'] + cc_flag)
                               }
            )
    ],
    cmdclass={
        'build_ext': BuildExtension
})

