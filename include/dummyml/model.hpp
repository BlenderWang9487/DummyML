#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace dummyml{

class Model
{
public:
    using nparray_d = py::array_t<double, py::array::c_style | py::array::forcecast> ;
    using nparray = py::array;
    virtual void load(const char*) = 0;
    virtual void save(const char*) = 0;
    virtual void fit(nparray_d, nparray_d) = 0;
    virtual nparray_d operator()(nparray_d) = 0;
};

}