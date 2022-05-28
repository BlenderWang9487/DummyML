#include <k_means.hpp>

void export_k_means(py::module_ &m){
    py::class_<dummyml::k_means>(m, "k_means")
        .def(py::init<>())
        .def(py::init<const char*>())
        .def(py::init<int,int>())
        .def("load",&dummyml::k_means::load)
        .def("save",&dummyml::k_means::save)
        .def("fit",&dummyml::k_means::fit_clusters)
        .def("__call__",
            [](
                dummyml::k_means& kms,
                dummyml::Model::nparray_d arr
            ){
                return kms.pred(arr);
            }
        )
        .def_property_readonly("inertia",[](dummyml::k_means& kms)->double{
            return kms.inertia();
        })
        .def("means",&dummyml::k_means::means)
        ;
}