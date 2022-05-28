#pragma once
#include <model.hpp>
#include <utils.hpp>
#include <cstdlib>
#include <ctime>

namespace dummyml
{

class dummy_clustering : public Model
{
private:
    size_t _k;
public:
    dummy_clustering(size_t k = 2): _k(k){
        srand(time(NULL));
    }
    void load(const char*){}
    void save(const char*){}
    void fit(nparray_d, nparray_d){}
    void fit_clusters(nparray_i y){
        auto y_mod = y.mutable_unchecked<1>();
        for(size_t row = 0;row < y_mod.shape(0);++row)
            y_mod(row) = rand() % _k;
        return;
    }
    nparray_d operator()(nparray_d){
        return nparray_d();
    }
    int pred(nparray_d x){
        return rand() % _k;
    }
    inline size_t get_k() const{
        return _k;
    }
};

} // namespace dummyml

void export_dummy_clustering(py::module_ &);