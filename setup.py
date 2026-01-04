from setuptools import setup, Extension
import pybind11


cpp_module = Extension(
    'data_driven_module',
    sources=['src/data_driven_space_filling_curve.cpp'],
    include_dirs=[pybind11.get_include()],
    extra_compile_args = ["-std=c++20"],
    language='c++'
)


setup(
    name='data_driven_module',
    version='1.0',
    description='Python package with C++ extension',
    ext_modules=[cpp_module],
)