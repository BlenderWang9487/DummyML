#include <k_means.hpp>

void export_k_means(py::module_ &m){
    py::class_<dummyml::k_means>(m, "k_means")
        .def(py::init<int,int,dummyml::Model::nparray_d>())
        .def("load",&dummyml::k_means::load)
        .def("save",&dummyml::k_means::save)
        .def("fit",&dummyml::k_means::fit)
        .def("__call__",
            [](
                dummyml::k_means& kms,
                dummyml::Model::nparray_d arr
            ){
                return kms(arr);
            }
        )
        .def("initialize_means",&dummyml::k_means::initialize_means)
        .def("means",&dummyml::k_means::means)
        .def("sum_of_distance",&dummyml::k_means::sum_of_distance)
        ;
}