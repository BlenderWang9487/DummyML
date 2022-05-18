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
    k_means(size_t feature_size, size_t k, nparray_d init_means):
        _feature_size(feature_size),
        _k(k),
        _means(k, std::vector<double>(feature_size)){
        initialize_means(init_means);
    }
    void initialize_means(nparray_d& init_means){
        if(init_means.size() != _k*_feature_size){
            throw std::length_error(
                "[ERROR] k_means initialize_means: means size mismatch."
            );
        }
        for(size_t k = 0;k < _k;++k)
            for(size_t f = 0;f < _feature_size;++f)
                _means[k][f] = init_means.at(k*_feature_size + f);
    }
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
        
        // assign cluster to every datapoint (E step)
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

        // calculate each cluster's new mean (M step)
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
    double sum_of_distance(nparray_d x) const {
        auto x_buf_info = x.request();
        if(x_buf_info.shape[1] != _feature_size){
            throw std::length_error(
                "[ERROR] k_means sum_of_distance: feature size mismatch."
            );
        }
        double sum_of_dis = 0.0;
        size_t dataset_size = x_buf_info.shape[0];
        double* x_ptr = (double*)x_buf_info.ptr;
        for(size_t index = 0;index < dataset_size;++index){
            double min_distance = std::numeric_limits<double>::infinity();
            for(size_t cluster = 0;cluster < _k;++cluster){
                double distance = 0.0;
                for(size_t feature = 0;feature < _feature_size;++feature){
                    double dif =
                        x_ptr[index * _feature_size + feature] -
                        _means[cluster][feature];
                    distance += dif*dif;
                }
                min_distance = std::min(min_distance, distance);
            }
            sum_of_dis += min_distance;
        }
        return sum_of_dis;
    }
};

} // namespace dummyml

void export_k_means(py::module_ &);