#pragma once
#include <model.hpp>
#include <utils.hpp>
#include <exception>
#include <vector>
#include <iostream>
#include <fstream>

namespace dummyml
{

class k_means : public Model
{
private:
    size_t _k;
    size_t _feature_size;
    double _inertia = 0.0;
    std::vector<std::vector<double>> _means;
public:
    struct MetaData{
        size_t _k;
        size_t _f;
        size_t check_sum;
        MetaData() = default;
        MetaData(const k_means& model):
            _k(model._k),
            _f(model._feature_size),
            check_sum(
                model._k ^
                model._feature_size
            ){}
        bool is_check_sum_correct(){
            return (_k ^ _f) == check_sum;
        }
    };
    k_means() = default;
    k_means(const char* file_name){
        load(file_name);
    }
    k_means(size_t feature_size, size_t k):
        _feature_size(feature_size),
        _k(k),
        _means(k, std::vector<double>(feature_size)){}
    void load(const char* file_name){
        std::fstream fin_bin(file_name,std::ios_base::in | std::ios_base::binary);
        if(fin_bin.fail()){
            throw std::runtime_error(
                "[ERROR] k_means load: failed to load model."
            );
        }
        MetaData meta;
        fin_bin.read(
            dummy_cast<char*,MetaData*>(&meta),
            sizeof(MetaData)
        );
        if(!meta.is_check_sum_correct()){
            throw std::runtime_error(
                "[ERROR] k_means load: MetaData mismatch."
            );
        }
        _k = meta._k;
        _feature_size = meta._f;
        _means = std::vector<std::vector<double>>(
            _k,
            std::vector<double>(_feature_size)
        );
        for(size_t k = 0;k < _k;++k)
            fin_bin.read(
                dummy_cast<char*,double*>(_means[k].data()),
                sizeof(double) * _means[k].size()
            );
        return;
    }
    void save(const char* file_name){
        std::fstream fout_bin(file_name,std::ios_base::out | std::ios_base::binary);
        if(fout_bin.fail()){
            throw std::runtime_error(
                "[ERROR] k_means save: failed to save model."
            );
        }
        MetaData meta(*this);
        fout_bin.write(
            dummy_cast<char*,MetaData*>(&meta),
            sizeof(MetaData)
        );
        for(size_t k = 0;k < _k;++k)
            fout_bin.write(
                dummy_cast<char*,double*>(_means[k].data()),
                sizeof(double) * _means[k].size()
            );
        return;
    }
    void fit(nparray_d, nparray_d){}
    void fit_clusters(nparray_d x, nparray_i y){
        // each y will be assigned a cluster after fitting.
        auto x_mod = x.unchecked<2>();
        double* x_ptr = (double*)x.request().ptr;
        auto y_mod = y.mutable_unchecked<1>();
        if(x_mod.shape(1) != _feature_size){
            throw std::length_error(
                "[ERROR] k_means fit: feature size mismatch."
            );
        }
        if(x_mod.shape(0) != y_mod.shape(0)){
            throw std::length_error(
                "[ERROR] k_means fit: data & cluster counts mismatch."
            );
        }
        size_t dataset_size = y_mod.shape(0);
        
        // calculate each cluster's new mean
        std::vector<size_t> count(_k);
        _means.assign(_k,std::vector<double>(_feature_size));
        for(size_t index = 0;index < dataset_size;++index){
            int cluster = y_mod(index);
            ++count[cluster];
            for(size_t feature = 0;feature < _feature_size;++feature)
                _means[cluster][feature] += x_ptr[index * _feature_size + feature];
        }
        for(size_t cluster = 0;cluster < _k;++cluster)
            if(count[cluster])
                for(size_t feature = 0;feature < _feature_size;++feature)
                    _means[cluster][feature] /= count[cluster];
        
        // assign cluster to every datapoint
        _inertia = 0.0;
        for(size_t index = 0;index < dataset_size;++index){
            double min_distance = std::numeric_limits<double>::infinity();
            int min_cluster = 0;
            for(int cluster = 0;cluster < _k;++cluster){
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
            y_mod(index) = min_cluster;
            _inertia += min_distance;
        }
    }
    nparray_d operator()(nparray_d){
        return nparray_d();
    }
    int pred(nparray_d x){
        auto x_mod = x.unchecked<1>();
        if(x_mod.shape(0) != _feature_size){
            throw std::length_error(
                "[ERROR] k_means(): feature size mismatch."
            );
        }

        double min_distance = std::numeric_limits<double>::infinity();
        size_t min_cluster = 0.0;
        for(size_t cluster = 0;cluster < _k;++cluster){
            double distance = 0.0;
            for(size_t feature = 0;feature < _feature_size;++feature){
                double dif = x_mod(feature) - _means[cluster][feature];
                distance += dif*dif;
            }
            if(distance < min_distance){
                min_distance = distance;
                min_cluster = cluster;
            }
        }
        return min_cluster;
    }
    nparray_d means() const {
        nparray_d ms({_k, _feature_size});
        auto ms_ptr = (double*)ms.request().ptr;
        for(size_t k = 0;k < _k;k++)
            memcpy(
                ms_ptr + k * _feature_size,
                _means[k].data(),
                _feature_size * sizeof(double)
            );
        return ms;
    }
    inline double inertia() const {
        return _inertia;
    }
};

} // namespace dummyml

void export_k_means(py::module_ &);