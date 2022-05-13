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
    double _alpha;
    std::unique_ptr<kernel> _k;
public:
    gaussian_process(
        double alpha = 5,
        std::unique_ptr<kernel> k = std::make_unique<linear_kernel>()
    ): _alpha(alpha), _k(){}
    gaussian_process(
        nparray_d x,
        nparray_d y,
        double alpha = 5,
        std::unique_ptr<kernel> k = std::make_unique<linear_kernel>()
    ): _alpha(alpha), _k(std::move(k)){
        fit(x, y);
    }
    void load(const char*){
        
    }
    void save(const char*){
        
    }
    void fit(nparray_d x, nparray_d y){
        
    }
    nparray_d operator()(nparray_d x){

    }
    void set_alpha(double alpha){
        _alpha = alpha;
    }
};

} // namespace dummyml

void export_gaussian_process(py::module_ &);