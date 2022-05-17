#pragma once
#include <model.hpp>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
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
    Eigen::MatrixXd _x;
    Eigen::MatrixXd _y;
    Eigen::MatrixXd _C_inv;
    double _alpha;
    std::unique_ptr<kernel> _k;
public:
    gaussian_process(
        double alpha = 0.2,
        kernel::type k_type = kernel::type::LinearKernel
    ): _alpha(alpha), _k(get_kernel(k_type)){}
    gaussian_process(
        nparray_d x,
        nparray_d y,
        double alpha = 0.2,
        kernel::type k_type = kernel::type::LinearKernel
    ): _alpha(alpha), _k(get_kernel(k_type)){
        fit(x, y);
    }
    void load(const char*){
        
    }
    void save(const char*){
        
    }
    void fit(nparray_d x, nparray_d y){
        auto x_buf_info = x.request();
        auto y_buf_info = y.request();
        if(x_buf_info.shape[0] != y_buf_info.shape[0]){
            throw std::length_error(
                "[ERROR] gaussian_process fit: data & label counts mismatch."
            );
        }
        size_t dataset_size = x_buf_info.shape[0];
        size_t feature_size = x_buf_info.shape[1];
        double* x_ptr = (double*)x_buf_info.ptr;
        double* y_ptr = (double*)y_buf_info.ptr;
        
        // copy x,y to _x,_y
        _x.resize(dataset_size, feature_size);
        _y.resize(dataset_size, 1);
        _C_inv.resize(dataset_size, dataset_size);
        memcpy(_x.data(), x_ptr, dataset_size*feature_size);
        memcpy(_y.data(), y_ptr, dataset_size);
        
        // calculate _C_inv
        for(size_t row = 0;row < dataset_size; ++row)
            for(size_t col = 0;col < dataset_size; ++col)
                _C_inv(row, col) = (*_k)(_x.row(row), _x.row(col));
        for(size_t diag = 0;diag < dataset_size; ++diag)
            _C_inv(diag ,diag) += _alpha;
        _C_inv = _C_inv.inverse();
    }
    nparray_d operator()(nparray_d x){
        auto x_buf_info = x.request();
        size_t dataset_size = _x.rows();
        size_t feature_size = _x.cols();
        if(x_buf_info.shape[0] != feature_size){
            throw std::length_error(
                "[ERROR] gaussian_process operator(): x size & feature size mismatch."
            );
        }
        Eigen::VectorXd x_vec(feature_size);
        Eigen::MatrixXd kt_vec(1, dataset_size);
        memcpy(x_vec.data(), x_buf_info.ptr, _x.cols());

        for(size_t i = 0;i < dataset_size;++i)
            kt_vec(0, i) = (*_k)(_x.row(i), x_vec);
        Eigen::MatrixXd ktC_inv = kt_vec * _C_inv;
        nparray_d result(2);
        double* result_ptr = (double*)result.request().ptr;
        result_ptr[0] = (ktC_inv * _y)(0);
        result_ptr[1] =
            ((*_k)(x_vec, x_vec) + _alpha) - // c
            (ktC_inv * kt_vec.transpose())(0);            // kt * C^-1 * k
        return result;
    }
    void set_alpha(double alpha){
        _alpha = alpha;
    }
};

} // namespace dummyml

void export_gaussian_process(py::module_ &);