#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace dummyml{

class Model
{
public:
    typedef py::array_t<double, py::array::c_style | py::array::forcecast> nparray;
    virtual void load(const char*) = 0;
    virtual void save(const char*) = 0;
    virtual void fit(nparray) = 0;
    virtual nparray operator()(nparray) = 0;
};

}