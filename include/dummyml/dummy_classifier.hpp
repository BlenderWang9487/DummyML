#pragma once
#include <model.hpp>
#include <iostream>
namespace dummyml
{

class dummy_classifier : public Model
{
private:
    size_t _feature_size;
    size_t _class_number;
public:
    dummy_classifier (size_t feature_size, size_t class_number):
    _feature_size(feature_size), _class_number(class_number){
        if(_class_number == 0){
            std::cout<<"[WARNING] dummy_classifier Ctr: _class_number can't be zero, forced changed to 1."<<std::endl;
            _class_number = 1;
        }
    }
    void load(const char*){
        std::cout<<"[NOTE] dummy_classifier load(): This func does nothing."<<std::endl;
        return;
    }
    void save(const char*){
        std::cout<<"[NOTE] dummy_classifier save(): This func does nothing."<<std::endl;
        return;
    }
    void fit(nparray){
        std::cout<<"[NOTE] dummy_classifier fit(): This func does nothing."<<std::endl;
        return;    
    }
    nparray operator()(nparray arr){
        auto result = nparray(_class_number);
        auto buf_info = result.request();
        double* ptr = (double*)buf_info.ptr;
        ptr[0] = 1.0;
        for(size_t i = 1;i < buf_info.size; i++)
            ptr[i] = 0.0;
        return result;
    }
};

}

void export_dummy_classifier(py::module_ &);