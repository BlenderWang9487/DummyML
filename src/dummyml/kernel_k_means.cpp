#include <kernel_k_means.hpp>

void export_kernel_k_means(py::module_ &m){
    py::class_<dummyml::kernel_k_means>(m, "kernel_k_means")
        .def(py::init<
                size_t,
                dummyml::Model::nparray_d,
                dummyml::kernel::type
            >(),
            py::arg("k"),
            py::arg("x"),
            py::arg("k_type") = dummyml::kernel::type::LinearKernel
        )
        .def("load",&dummyml::kernel_k_means::load)
        .def("save",&dummyml::kernel_k_means::save)
        .def("fit",&dummyml::kernel_k_means::fit_clusters)
        .def("run_kernel",&dummyml::kernel_k_means::run_kernel) // debug
        .def_property_readonly("inertia",&dummyml::kernel_k_means::inertia)
        ;
}