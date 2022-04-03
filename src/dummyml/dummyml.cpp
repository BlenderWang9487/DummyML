#include <pybind11/pybind11.h>
#include <dummy_classifier.hpp>
#include <dummy_regression.hpp>
#include <utils.hpp>

namespace py = pybind11;

PYBIND11_MODULE(dummyml, m){
    export_dummy_classifier(m);
    export_dummy_regression(m);
}