#include <dummy_clustering.hpp>

void export_dummy_clustering(py::module_ &m){
    py::class_<dummyml::dummy_clustering>(m, "dummy_clustering")
        .def(py::init<size_t>(), py::arg("k") = 2)
        .def("load",&dummyml::dummy_clustering::load)
        .def("save",&dummyml::dummy_clustering::save)
        .def("fit",&dummyml::dummy_clustering::fit_clusters)
        .def("__call__",
            [](
                dummyml::dummy_clustering& dctr,
                dummyml::Model::nparray_d arr
            ){
                return dctr.pred(arr);
            }
        )
        .def_property_readonly("k",&dummyml::dummy_clustering::get_k)
        ;
}