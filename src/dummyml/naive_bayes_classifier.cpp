#include <naive_bayes_classifier.hpp>

void export_naive_bayes_classifier(py::module_ &m){
    py::class_<dummyml::naive_bayes_classifier>(m, "naive_bayes_classifier")
        .def(py::init<>())
        .def(py::init<const char*>())
        .def(py::init<int,int>())
        .def("load",&dummyml::naive_bayes_classifier::load)
        .def("save",&dummyml::naive_bayes_classifier::save)
        .def("fit",&dummyml::naive_bayes_classifier::fit)
        .def("__call__",
            [](
                dummyml::naive_bayes_classifier& nbclsfr,
                dummyml::Model::nparray_d arr
            ){
                return nbclsfr(arr);
            }
        )
        ;
}