#pragma once
#include <model.hpp>
#include <utils.hpp>
#include <exception>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

namespace dummyml
{

class k_means : public Model
{
private:
    size_t _k;
    size_t _feature_size;
    std::vector<std::vector<double>> _means;
public:
    k_means(size_t feature_size,size_t k):
        _feature_size(feature_size),
        _k(k),
        _means(k, std::vector<double>(feature_size)){
        srand(time(nullptr));
        
        for(auto& mean:_means)
            for(auto& per_dim:mean)
                per_dim = (rand() / (double)RAND_MAX) * 100000.0 - 50000.0;
    }
    void load(const char*){
        
    }
    void save(const char*){
        
    }
    void fit(nparray_d x, nparray_d y){
        // each y will be assigned a cluster after fitting.
        auto x_buf_info = x.request();
        auto y_buf_info = y.request();
        if(x_buf_info.shape[1] != _feature_size){
            throw std::length_error(
                "[ERROR] k_means fit: feature size mismatch."
            );
        }
        if(x_buf_info.shape[0] != y_buf_info.shape[0]){
            throw std::length_error(
                "[ERROR] k_means fit: data & cluster counts mismatch."
            );
        }
        if(y_buf_info.shape[1] != 1){
            throw std::length_error(
                "[ERROR] k_means fit: each data in y only need one number."
            );
        }
        size_t dataset_size = y_buf_info.shape[0];
        double* x_ptr = (double*)x_buf_info.ptr;
        double* y_ptr = (double*)y_buf_info.ptr;
        
        // assign cluster to every datapoint
        for(size_t index = 0;index < dataset_size;++index){
            double min_distance = std::numeric_limits<double>::infinity();
            size_t min_cluster = 0.0;
            for(size_t cluster = 0;cluster < _k;++cluster){
                double distance = 0.0;
                for(size_t feature = 0;feature < _feature_size;++feature){
                    double dif =
                        x_ptr[index * _feature_size + feature] -
                        _means[cluster][feature];
                    distance += dif*dif;
                }
                if(distance < min_distance){
                    min_distance = distance;
                    min_cluster = cluster;
                }
            }
            y_ptr[index] = (double) min_cluster;
        }

        // calculate each cluster's new mean
        std::vector<size_t> count(_k);
        _means.assign(_k,std::vector<double>(_feature_size));
        for(size_t index = 0;index < dataset_size;++index){
            size_t cluster = (size_t)std::round(y_ptr[index]);
            ++count[cluster];
            for(size_t feature = 0;feature < _feature_size;++feature)
                _means[cluster][feature] += x_ptr[index * _feature_size + feature];
        }
        for(size_t cluster = 0;cluster < _k;++cluster)
            if(count[cluster])
                for(size_t feature = 0;feature < _feature_size;++feature)
                    _means[cluster][feature] /= count[cluster];
    }
    nparray_d operator()(nparray_d x){
        if(x.size() != _feature_size){
            throw std::length_error(
                "[ERROR] k_means(): feature size mismatch."
            );
        }

        double min_distance = std::numeric_limits<double>::infinity();
        size_t min_cluster = 0.0;
        for(size_t cluster = 0;cluster < _k;++cluster){
            double distance = 0.0;
            for(size_t feature = 0;feature < _feature_size;++feature){
                double dif =
                   x.at(feature) -
                    _means[cluster][feature];
                distance += dif*dif;
            }
            if(distance < min_distance){
                min_distance = distance;
                min_cluster = cluster;
            }
        }
        nparray_d result(1);
        *(double*)result.request().ptr = (double)min_cluster;
        return result;
    }
};

} // namespace dummyml

void export_k_means(py::module_ &);