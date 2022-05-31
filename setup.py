import sys
from glob import glob
DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(DIR, "extern", "pybind11"))
from pybind11.setup_helpers import Pybind11Extension, build_ext, ParallelCompile
import subprocess
import os
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cwd = Path().absolute()

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        pyenv_root = os.environ.get("PYENV_ROOT")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        if pyenv_root is not None:
            cmake_args += [f"-DPYTHON_EXECUTABLE={pyenv_root}/shims/python"]

        build_args = ["--config", "Release", "--", "-j4"]

        env = os.environ.copy()

        self.announce(f"Running CMake prepare", level=3)
        subprocess.check_call(["cmake", cwd] + cmake_args, cwd=self.build_temp, env=env)

        self.announce("Building extensions")
        cmake_cmd = ["cmake", "--build", "."] + build_args
        subprocess.check_call(cmake_cmd, cwd=self.build_temp)


ext_modules = [
    Pybind11Extension(
        'dummyml',
        glob('src/dummyml/*.cpp'),
        include_dirs=[
            "include/dummyml",
            "extern/eigen"
        ],
        cxx_std=17,
        include_pybind11=True
    )
]

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

setup(
    name="dummyml",
    version="1.0.0",
    author="BlenderWang9487",
    author_email="developinblend@gmail.com",
    description="A Dummy ML library",
    long_description=
"""
An easy-to-use ML library for people new to ml,
APIs are provided in pthon3,
core functions are implemented in C++
""",
    zip_safe=False,
    license="MIT",
    install_requires=[
        "numpy>=1.7.0"
    ],
    cmdclass=dict(build_ext=build_ext),
    ext_modules=ext_modules,
)