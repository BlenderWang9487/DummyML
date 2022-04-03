#pragma once
#include <model.hpp>

namespace dummyml
{

class dummy_classifier : public Model
{
private:
    size_t _feature_size;
    size_t _class_number;
public:
    dummy_classifier (size_t feature_size, size_t class_number):
        _feature_size(feature_size), _class_number(class_number){};
    void load(const char*){
        return;
    }
    void save(const char*){
        return;
    }
    void fit(nparray){
        return;    
    }
    nparray operator()(nparray arr){
        double* dummy_output = new double[_class_number]();
        *(dummy_output) = 1.0;
        return nparray_wrapper(_class_number,dummy_output);
    }
};

}

void export_dummy_classifier(py::module_ &);