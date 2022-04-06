#pragma once
#include <model.hpp>
#include <utils.hpp>
#include <exception>
#include <iostream>
#include <fstream>

namespace dummyml
{

class naive_bayes_classifier : public Model
{
private:
    size_t _accumulation = 0;
    size_t _feature_size;
    size_t _class_number;
    mean_variance likelihood;
    mean_variance marginal;
    std::vector<size_t> prior;
public:
    naive_bayes_classifier (size_t feature_size, size_t class_number):
    _feature_size(feature_size), _class_number(class_number){
        if(_class_number == 0){
            std::cout<<"[WARNING] naive_bayes_classifier Ctr: _class_number can't be zero, forced changed to 1."<<std::endl;
            _class_number = 1;
        }
        likelihood = mean_variance(_feature_size * class_number);
        marginal = mean_variance(_feature_size);
        prior = std::vector<size_t>(_class_number);
    }
    void load(const char*){
        /* TODO
        1. load data to likelihood and marginal table
        2. if failed, throw exception
        */
        return;
    }
    void save(const char*){
        /* TODO
        1. save likelihood and marginal table to file
        2. if failed, throw exception
        */
        return;
    }
    void fit(nparray_d x, nparray_d y) {
        // ***using Welford's online algorithm to calc mean & variance
        // if I find out it's too slow, I'll change it.
        auto x_buf_info = x.request();
        auto y_buf_info = y.request();
        if(x_buf_info.shape[1] != _feature_size){
            throw std::length_error(
                "[ERROR] naive_bayes_classifier fit: feature size mismatch."
            );
        }
        if(x_buf_info.shape[0] != y_buf_info.shape[0]){
            throw std::length_error(
                "[ERROR] naive_bayes_classifier fit: data & label counts mismatch."
            );
        }
        if(y_buf_info.shape[1] != _class_number){
            throw std::length_error(
                "[ERROR] naive_bayes_classifier fit: class num mismatch."
            );
        }
        size_t dataset_size = y_buf_info.shape[0];
        double* x_ptr = (double*)x_buf_info.ptr;
        double* y_ptr = (double*)y_buf_info.ptr;
        for(size_t i = 0; i < dataset_size; ++i){
            // get y_label from one-hot
            size_t y_label = 0;
            double* y_ptr_current = y_ptr + i*_class_number;
            for(size_t j = 0; j < _class_number;++j)
                if(y_ptr_current[y_label] < y_ptr_current[j])
                    y_label = j;
            // update prior **NOTE: prior is int should divided by _accumulation when inferencing
            ++prior[y_label];
            ++_accumulation;
            for(size_t j = 0; j < _feature_size; ++j){
                double x_n = x_ptr[i*_feature_size + j];

                double old_like_mean = likelihood.mean(j*_class_number + y_label);
                double old_marg_mean = marginal.mean(j);
                
                likelihood.mean(j*_class_number + y_label) += (x_n - old_like_mean) / _accumulation;
                marginal.mean(j) += (x_n - old_marg_mean) / _accumulation;

                likelihood.variance(j*_class_number + y_label) += 
                    ((x_n - old_like_mean)*(x_n - likelihood.mean(j*_class_number + y_label)) -
                    likelihood.variance(j*_class_number + y_label)) / _accumulation;
                marginal.variance(j) +=
                    ((x_n - old_marg_mean)*(x_n - marginal.mean(j)) -
                    marginal.variance(j)) / _accumulation;
            }
        }
        return;    
    }
    nparray_d operator()(nparray_d x){
        if(x.size() != _feature_size){
            throw std::length_error(
                "[ERROR] naive_bayes_classifier(): feature size mismatch."
            );
        }
        std::vector<double> result_vec(_class_number);
        for(size_t i = 0;i < _class_number;++i){
            double posterior = log((double)prior[i]/_accumulation);
            for(size_t j = 0;j < _feature_size;j++){
                posterior += mean_variance::logNormalDistribution(
                    likelihood.mean(j*_class_number + i),
                    std::max(likelihood.variance(j*_class_number + i), 0.2),
                    x.at(j),
                    true) - 
                    mean_variance::logNormalDistribution(
                    marginal.mean(j),
                    std::max(marginal.variance(j), 0.2),
                    x.at(j),
                    true);
            }
            result_vec[i] = posterior;
        }
        auto softmax_vec = softmax(result_vec); 
        nparray_d result(_class_number);
        std::memcpy(result.request().ptr,softmax_vec.data(),softmax_vec.size()*sizeof(double));
        return result;
    }
};

}

void export_naive_bayes_classifier(py::module_ &);