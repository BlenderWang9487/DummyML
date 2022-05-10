#pragma once
#include <model.hpp>
#include <utils.hpp>
#include <exception>
#include <vector>
#include <iostream>
#include <fstream>

namespace dummyml
{

class gaussian_process : public Model
{
private:
    Eigen::MatrixXd _x_train;
    Eigen::MatrixXd _C;
    kernel* _k;
public:
    gaussian_process() = default;
    void load(const char*){
        
    }
    void save(const char*){
        
    }
    void fit(nparray_d x, nparray_d y){
        
    }
    nparray_d operator()(nparray_d x){

    }
};

} // namespace dummyml

void export_gaussian_process(py::module_ &);