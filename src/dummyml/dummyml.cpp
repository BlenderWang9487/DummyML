#include <pybind11/pybind11.h>
#include <dummy_classifier.hpp>
#include <dummy_regression.hpp>
#include <naive_bayes_classifier.hpp>
#include <k_means.hpp>
#include <gaussian_process.hpp>
#include <utils.hpp>

namespace py = pybind11;

PYBIND11_MODULE(dummyml, m){
    export_dummy_classifier(m);
    export_dummy_regression(m);
    export_naive_bayes_classifier(m);
    export_k_means(m);
    export_gaussian_process(m);
    export_utils(m);
}