#include <pybind11/pybind11.h>
#include <dummy_classifier.hpp>
#include <utils.hpp>

namespace py = pybind11;

PYBIND11_MODULE(dummyml, m){
    py::class_<dummyml::dummy_classifier>(m, "dummy_classifier")
        .def(py::init<int,int>())
        .def("load",&dummyml::dummy_classifier::load)
        .def("save",&dummyml::dummy_classifier::save)
        .def("fit",&dummyml::dummy_classifier::fit)
        .def("__call__",
            [](
                dummyml::dummy_classifier& dclsfr,
                dummyml::Model::nparray arr
            ){
                return dclsfr(arr);
            }
        )
        ;
}