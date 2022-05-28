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
    EigenMatrix _x;
    EigenMatrix _y;
    EigenMatrix _C_inv;
    double _alpha;
    std::unique_ptr<kernel> _kernel;
public:
    struct MetaData{
        Eigen::Index _n;
        Eigen::Index _f;
        kernel::type _t;
        double _a;
        Eigen::Index check_sum;
        MetaData() = default;
        MetaData(const gaussian_process& model):
            _n(model._x.rows()),
            _f(model._x.cols()),
            _t(model._kernel->_T),
            _a(model._alpha),
            check_sum(
                model._x.rows() ^
                model._x.cols() ^
                model._kernel->_T
            ){}
        bool is_check_sum_correct(){
            return (_n ^ _f ^ _t) == check_sum;
        }
    };
    gaussian_process(const char* filename){
        load(filename);
    }
    gaussian_process(
        double alpha = 0.2,
        kernel::type k_type = kernel::type::LinearKernel
    ): _alpha(alpha), _kernel(get_kernel(k_type)){}
    gaussian_process(
        nparray_d x,
        nparray_d y,
        double alpha = 0.2,
        kernel::type k_type = kernel::type::LinearKernel
    ): _alpha(alpha), _kernel(get_kernel(k_type)){
        fit(x, y);
    }
    double run_kernel(double x, double y){
        return (*_kernel)(x,y);
    }
    void load(const char* file_name){
        std::fstream fin_bin(file_name,std::ios_base::in | std::ios_base::binary);
        if(fin_bin.fail()){
            throw std::runtime_error(
                "[ERROR] gaussian_process load: failed to load model."
            );
        }
        MetaData meta;
        fin_bin.read(
            dummy_cast<char*,MetaData*>(&meta),
            sizeof(MetaData)
        );
        if(!meta.is_check_sum_correct()){
            throw std::runtime_error(
                "[ERROR] gaussian_process load: MetaData mismatch."
            );
        }
        _alpha = meta._a;
        _kernel = get_kernel(meta._t);
        _x = EigenMatrix(meta._n, meta._f);
        _y = EigenMatrix(meta._n, 1);
        _C_inv = EigenMatrix(meta._n, meta._n);
        fin_bin.read(
            dummy_cast<char*,double*>(_x.data()),
            sizeof(double) * _x.size()
        ).read(
            dummy_cast<char*,double*>(_y.data()),
            sizeof(double) * _y.size()
        ).read(
            dummy_cast<char*,double*>(_C_inv.data()),
            sizeof(double) * _C_inv.size()
        );
        return;
    }
    void save(const char* file_name){
        std::fstream fout_bin(file_name,std::ios_base::out | std::ios_base::binary);
        if(fout_bin.fail()){
            throw std::runtime_error(
                "[ERROR] gaussian_process save: failed to save model."
            );
        }
        MetaData meta(*this);
        fout_bin.write(
            dummy_cast<char*,MetaData*>(&meta),
            sizeof(MetaData)
        ).write(
            dummy_cast<char*,double*>(_x.data()),
            sizeof(double) * _x.size()
        ).write(
            dummy_cast<char*,double*>(_y.data()),
            sizeof(double) * _y.size()
        ).write(
            dummy_cast<char*,double*>(_C_inv.data()),
            sizeof(double) * _C_inv.size()
        );
        return;
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
        memcpy(_x.data(), x_ptr, dataset_size * feature_size * sizeof(double));
        memcpy(_y.data(), y_ptr, dataset_size                * sizeof(double));
        
        // calculate _C_inv
        for(size_t row = 0;row < dataset_size; ++row)
            for(size_t col = row;col < dataset_size; ++col)
                _C_inv(col, row) = _C_inv(row, col) = (*_kernel)(_x.row(row), _x.row(col));
        for(size_t diag = 0;diag < dataset_size; ++diag)
            _C_inv(diag ,diag) += _alpha;
        _C_inv = _C_inv.inverse();
        return;
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
        EigenVector x_vec(feature_size);
        EigenMatrix kt_vec(1, dataset_size);
        memcpy(x_vec.data(), x_buf_info.ptr, _x.cols()*sizeof(double));

        for(size_t i = 0;i < dataset_size;++i)
            kt_vec(0, i) = (*_kernel)(_x.row(i), x_vec);
        EigenMatrix ktC_inv = kt_vec * _C_inv;
        nparray_d result(2);
        double* result_ptr = (double*)result.request().ptr;
        result_ptr[0] = (ktC_inv * _y)(0);
        result_ptr[1] =
            ((*_kernel)(x_vec, x_vec) + _alpha) -  // c
            (ktC_inv * kt_vec.transpose())(0);// kt * C^-1 * k
        return result;
    }
    void set_alpha(double alpha){
        _alpha = alpha;
    }
};

} // namespace dummyml

void export_gaussian_process(py::module_ &);