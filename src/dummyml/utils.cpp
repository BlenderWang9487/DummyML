#include <utils.hpp>

std::vector<double> dummyml::softmax(const std::vector<double>& v){
    std::vector<double> ret(v.size());
    double sum = 0.0;
    for(size_t i = 0;i < v.size();++i)
        sum += ret[i] = exp(v[i]);
    for(size_t i = 0;i < v.size();++i)
        ret[i] /= sum;
    return ret;
}
