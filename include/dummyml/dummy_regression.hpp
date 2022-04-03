#pragma once
#include <model.hpp>
#include <iostream>
namespace dummyml
{

class dummy_regression : public Model
{
private:
    size_t _feature_size;
public:
    dummy_regression(size_t feature_size):
    _feature_size(feature_size){}
    void load(const char*){
        std::cout<<"[NOTE] dummy_regression load(): This func does nothing."<<std::endl;
        return;
    }
    void save(const char*){
        std::cout<<"[NOTE] dummy_regression save(): This func does nothing."<<std::endl;
        return;
    }
    void fit(nparray){
        std::cout<<"[NOTE] dummy_regression fit(): This func does nothing."<<std::endl;
        return;    
    }
    nparray operator()(nparray arr){
        auto result = nparray(1);
        auto buf_info = result.request();
        double* ptr = (double*)buf_info.ptr;
        ptr[0] = 1.0;
        return result;
    }
};

}

void export_dummy_regression(py::module_ &);