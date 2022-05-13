#include <gaussian_process.hpp>

void export_gaussian_process(py::module_ &m){
    py::class_<dummyml::gaussian_process>(m, "gaussian_process")
        .def(py::init<
            double,
            dummyml::kernel::type
        >())
        .def(py::init<
            dummyml::Model::nparray_d,
            dummyml::Model::nparray_d,
            double,
            dummyml::kernel::type
        >())
        .def("load",&dummyml::gaussian_process::load)
        .def("save",&dummyml::gaussian_process::save)
        .def("fit",&dummyml::gaussian_process::fit)
        .def("__call__",
            [](
                dummyml::gaussian_process& gp,
                dummyml::Model::nparray_d arr
            ){
                return gp(arr);
            }
        )
        ;
}