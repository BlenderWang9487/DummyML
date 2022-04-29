#pragma once
#include <model.hpp>
#include <utils.hpp>
#include <exception>
#include <vector>
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
    mean_variance _likelihood;
    mean_variance _marginal;
    std::vector<size_t> _prior;
public:
    struct MetaData{
        size_t _a;
        size_t _f;
        size_t _c;
        size_t check_sum;
        MetaData() = default;
        MetaData(const naive_bayes_classifier& model):
            _a(model._accumulation),
            _f(model._feature_size),
            _c(model._class_number),
            check_sum(
                model._accumulation ^
                model._feature_size ^
                model._class_number
            ){}
        bool is_check_sum_correct(){
            return (_a ^ _f ^ _c) == check_sum;
        }
    };
    naive_bayes_classifier() = default;
    naive_bayes_classifier (size_t feature_size, size_t class_number):
        _feature_size(feature_size),
        _class_number(class_number){
        if(_class_number == 0){
            std::cout<<"[WARNING] naive_bayes_classifier Ctr: _class_number can't be zero, forced changed to 1."<<std::endl;
            _class_number = 1;
        }
        _likelihood = mean_variance(_feature_size * _class_number);
        _marginal = mean_variance(_feature_size);
        _prior = std::vector<size_t>(_class_number);
    }
    void load(const char* file_name){
        /* TODO
        1. load data to likelihood and marginal table
        2. if failed, throw exception
        */
        std::fstream fin_bin(file_name,std::ios_base::in | std::ios_base::binary);
        if(fin_bin.fail()){
            throw std::runtime_error(
                "[ERROR] naive_bayes_classifier load: failed to load model."
            );
        }
        MetaData meta;
        fin_bin.read(
            dummy_cast<char*,MetaData*>(&meta),
            sizeof(MetaData)
        );
        if(!meta.is_check_sum_correct()){
            throw std::runtime_error(
                "[ERROR] naive_bayes_classifier load: MetaData mismatch."
            );
        }
        _accumulation = meta._a;
        _feature_size = meta._f;
        _class_number = meta._c;
        _likelihood = mean_variance(_feature_size * _class_number);
        _marginal = mean_variance(_feature_size);
        _prior = std::vector<size_t>(_class_number);

        fin_bin.read(
            dummy_cast<char*,double*>(_likelihood.mean_data()),
            sizeof(double) * _likelihood.size()
        ).read(
            dummy_cast<char*,double*>(_likelihood.variance_data()),
            sizeof(double) * _likelihood.size()
        ).read(
            dummy_cast<char*,double*>(_marginal.mean_data()),
            sizeof(double) * _marginal.size()
        ).read(
            dummy_cast<char*,double*>(_marginal.variance_data()),
            sizeof(double) * _marginal.size()
        ).read(
            dummy_cast<char*,size_t*>(_prior.data()),
            sizeof(size_t) * _prior.size()
        );
        return;
    }
    void save(const char* file_name){
        /* TODO
        1. save likelihood and marginal table to file
        2. if failed, throw exception
        */
        std::fstream fout_bin(file_name,std::ios_base::out | std::ios_base::binary);
        if(fout_bin.fail()){
            throw std::runtime_error(
                "[ERROR] naive_bayes_classifier save: failed to save model."
            );
        }
        MetaData meta(*this);
        fout_bin.write(
            dummy_cast<char*,MetaData*>(&meta),
            sizeof(MetaData)
        ).write(
            dummy_cast<char*,double*>(_likelihood.mean_data()),
            sizeof(double) * _likelihood.size()
        ).write(
            dummy_cast<char*,double*>(_likelihood.variance_data()),
            sizeof(double) * _likelihood.size()
        ).write(
            dummy_cast<char*,double*>(_marginal.mean_data()),
            sizeof(double) * _marginal.size()
        ).write(
            dummy_cast<char*,double*>(_marginal.variance_data()),
            sizeof(double) * _marginal.size()
        ).write(
            dummy_cast<char*,size_t*>(_prior.data()),
            sizeof(size_t) * _prior.size()
        );
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
            ++_prior[y_label];
            ++_accumulation;
            for(size_t j = 0; j < _feature_size; ++j){
                double x_n = x_ptr[i*_feature_size + j];
                size_t j_likelihood = j*_class_number + y_label;

                double old_like_mean = _likelihood.mean(j_likelihood);
                double old_marg_mean = _marginal.mean(j);
                
                _likelihood.mean(j_likelihood) += (x_n - old_like_mean) / _accumulation;
                _marginal.mean(j) += (x_n - old_marg_mean) / _accumulation;

                _likelihood.variance(j_likelihood) += 
                    ((x_n - old_like_mean)*(x_n - _likelihood.mean(j_likelihood)) -
                    _likelihood.variance(j_likelihood)) / _accumulation;
                _marginal.variance(j) +=
                    ((x_n - old_marg_mean)*(x_n - _marginal.mean(j)) -
                    _marginal.variance(j)) / _accumulation;
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
            double posterior = log((double)_prior[i]/_accumulation);
            for(size_t j = 0;j < _feature_size;++j){
                posterior += 
                    mean_variance::logNormalDistribution(
                    _likelihood.mean(j*_class_number + i),
                    std::max(_likelihood.variance(j*_class_number + i), 0.2),
                    x.at(j),
                    true) - 
                    mean_variance::logNormalDistribution(
                    _marginal.mean(j),
                    std::max(_marginal.variance(j), 0.2),
                    x.at(j),
                    true);
            }
            result_vec[i] = posterior;
        }
        auto softmax_vec = softmax(result_vec); 
        nparray_d result(_class_number);
        std::memcpy(
            result.request().ptr,
            softmax_vec.data(),
            softmax_vec.size()*sizeof(double)
        );
        return result;
    }
};

} // namespace dummyml

void export_naive_bayes_classifier(py::module_ &);