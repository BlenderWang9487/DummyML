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
    k_means(size_t feature_size,size_t k):
        _feature_size(feature_size),
        _k(k),
        _means(k, std::vector<double>(feature_size)){}
    void load(const char*){
        
    }
    void save(const char*){
        
    }
    void fit(nparray_d, nparray_d){
        
    }
    nparray_d operator()(nparray_d){

    }
};


} // namespace dummyml
