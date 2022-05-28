#pragma once

#include <model.hpp>
#include <utils.hpp>
#include <exception>
#include <vector>
#include <iostream>
#include <fstream>

namespace dummyml{

class kernel_k_means : public Model
{
private:
    size_t _k;
    bool is_large_dataset = false;
    double _inertia = std::numeric_limits<double>::infinity();;
    EigenMatrix _x;
    EigenMatrix _gram;
    std::unique_ptr<kernel> _kernel;
    void _space_consuming_approach(nparray_i& y){
        auto y_mod = y.mutable_unchecked<1>();
        size_t dataset_size = _x.rows();

        // calculate the Ck (count of each cluster)
        std::vector<int> Ck(_k);
        for(size_t cls = 0;cls < y_mod.shape(0);++cls)
            ++Ck[y_mod(cls)];
        for(size_t k = 0;k < _k;++k)
            Ck[k] = std::max(Ck[k], 1);

        // calculate the 2nd and 3rd term of square distance
        std::vector<double> k_pq(_k, 0.0);
        EigenMatrix k_jn(dataset_size, _k);
        k_jn.setZero();
        for(size_t row = 0;row < dataset_size;++row)
            for(size_t col = 0;col < dataset_size;++col){
                double k_rc = _gram(row, col);
                int clstr_col = y_mod(col);

                k_jn(row, clstr_col) += k_rc;

                // alpha_kp == alpha_kq == 1
                if(y_mod(row) == clstr_col)
                    k_pq[clstr_col] += k_rc;
            }
        for(size_t k = 0;k < _k;++k)
            k_pq[k] /= (double)(Ck[k] * Ck[k]);
        
        _inertia = 0.0;
        for(size_t row = 0;row < dataset_size;++row){
            for(size_t k = 0;k < _k;++k)
                k_jn(row, k) *= 2.0/Ck[k];
            
            // assign new cluster to each data
            // 'cause k(x_j, x_j) is constant for comparing the distance
            // so we ignore it
            int nearest_k = 0;
            double min_distance = std::numeric_limits<double>::infinity();
            for(size_t k = 0;k < _k;++k){
                double distance = -k_jn(row, k) + k_pq[k];
                if(distance < min_distance){
                    nearest_k = k;
                    min_distance = distance;
                }
            }
            _inertia += min_distance;
            y_mod(row) = nearest_k;
        }
        return;
    }
    void _space_efficient_approach(nparray_i& y){
        auto y_mod = y.mutable_unchecked<1>();
        size_t dataset_size = _x.rows();

        // calculate the Ck (count of each cluster)
        std::vector<int> Ck(_k);
        for(size_t cls = 0;cls < y_mod.shape(0);++cls)
            ++Ck[y_mod(cls)];
        for(size_t k = 0;k < _k;++k)
            Ck[k] = std::max(Ck[k], 1);

        // calculate the 2nd and 3rd term of square distance
        std::vector<double> k_pq(_k, 0.0);
        EigenMatrix k_jn(dataset_size, _k);
        k_jn.setZero();
        for(size_t row = 0;row < dataset_size;++row)
            for(size_t col = 0;col < dataset_size;++col){
                double k_rc = (*_kernel)(_x.row(row), _x.row(col));
                int clstr_col = y_mod(col);

                k_jn(row, clstr_col) += k_rc;

                // alpha_kp == alpha_kq == 1
                if(y_mod(row) == clstr_col)
                    k_pq[clstr_col] += k_rc;
            }
        for(size_t k = 0;k < _k;++k)
            k_pq[k] /= (double)(Ck[k] * Ck[k]);
        
        _inertia = 0.0;
        for(size_t row = 0;row < dataset_size;++row){
            for(size_t k = 0;k < _k;++k)
                k_jn(row, k) *= 2.0/Ck[k];
            
            // assign new cluster to each data
            // 'cause k(x_j, x_j) is constant for comparing the distance
            // so we ignore it
            int nearest_k = 0;
            double min_distance = std::numeric_limits<double>::infinity();
            for(size_t k = 0;k < _k;++k){
                double distance = -k_jn(row, k) + k_pq[k];
                if(distance < min_distance){
                    nearest_k = k;
                    min_distance = distance;
                }
            }
            _inertia += min_distance;
            y_mod(row) = nearest_k;
        }
        return;
    }
public:
    kernel_k_means() = default;
    kernel_k_means(
        size_t k,
        nparray_d x,
        kernel::type k_type = kernel::type::LinearKernel
    ): _k(k), _kernel(get_kernel(k_type)){
        auto x_buf_info = x.request();
        if(x_buf_info.ndim != 2){
            throw std::length_error(
                "[ERROR] kernel_k_means Ctor: x wrong dimension."
            );
        }
        size_t dataset_size = x_buf_info.shape[0];
        size_t feature_size = x_buf_info.shape[1];
        _x.resize(dataset_size, feature_size);
        memcpy(
            _x.data(),
            x_buf_info.ptr,
            dataset_size * feature_size * sizeof(double)
        );

        if(dataset_size > 10000){
            is_large_dataset = true;
        }else{
            is_large_dataset = false;
            // compute gram matrix
            _gram.resize(dataset_size, dataset_size);
            for(size_t row = 0;row < dataset_size; ++row)
                for(size_t col = row;col < dataset_size; ++col)
                    _gram(col, row) = _gram(row, col) = (*_kernel)(_x.row(row), _x.row(col));
        }
    }
    double run_kernel(double x, double y){
        return (*_kernel)(x,y);
    }
    void load(const char* file_name){
        // TODO
    }
    void save(const char* file_name){
        // TODO
    }
    void fit(nparray_d, nparray_d){}
    void fit_clusters(nparray_i y){
        auto y_mod = y.mutable_unchecked<1>();
        size_t dataset_size = _x.rows();
        if(y_mod.shape(0) != dataset_size){
            throw std::length_error(
                "[ERROR] kernel_k_means fit: data & label counts mismatch."
            );
        }
        
        if(is_large_dataset)
            _space_efficient_approach(y);
        else
            _space_consuming_approach(y);
    }
    nparray_d operator()(nparray_d){
        return nparray_d();
    }
    inline double inertia() const{
        return _inertia;
    }
};

} // namespace dummyml

void export_kernel_k_means(py::module_ &);