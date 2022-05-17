#include <gaussian_process.hpp>

void export_gaussian_process(py::module_ &m){
    py::class_<dummyml::gaussian_process>(m, "gaussian_process")
        .def(
            py::init<
                double,
                dummyml::kernel::type
            >(),
            py::arg("alpha") = 0.2,
            py::arg("k_type") = dummyml::kernel::type::LinearKernel)
        .def(
            py::init<
                dummyml::Model::nparray_d,
                dummyml::Model::nparray_d,
                double,
                dummyml::kernel::type
            >(),
            py::arg("x"),
            py::arg("y"),
            py::arg("alpha") = 0.2,
            py::arg("k_type") = dummyml::kernel::type::LinearKernel
            )
        .def("load",&dummyml::gaussian_process::load)
        .def("save",&dummyml::gaussian_process::save)
        .def(
            "fit",
            &dummyml::gaussian_process::fit,
            py::arg("x"),
            py::arg("y")
        )
        .def("__call__",
            [](
                dummyml::gaussian_process& gp,
                dummyml::Model::nparray_d arr
            ){
                return gp(arr);
            }
        )
        .def("run_kernel",&dummyml::gaussian_process::run_kernel) // debug
        ;
}