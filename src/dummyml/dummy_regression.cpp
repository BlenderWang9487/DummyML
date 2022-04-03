#include <dummy_regression.hpp>

void export_dummy_regression(py::module_ &m){
    py::class_<dummyml::dummy_regression>(m, "dummy_regression")
        .def(py::init<int>())
        .def("load",&dummyml::dummy_regression::load)
        .def("save",&dummyml::dummy_regression::save)
        .def("fit",&dummyml::dummy_regression::fit)
        .def("__call__",
            [](
                dummyml::dummy_regression& drgsn,
                dummyml::Model::nparray arr
            ){
                return drgsn(arr);
            }
        )
        ;
}