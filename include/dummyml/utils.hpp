#pragma once
#include <vector>
#include <cmath>
namespace dummyml{

class mean_variance
{
private:
    std::vector<double> _mean;
    std::vector<double> _variance;
public:
    mean_variance() = default;
    mean_variance(size_t size): _mean(size), _variance(size){}
    inline size_t size() const {
        return _mean.size();
    }
    inline const    double& mean    (size_t index) const {
        return _mean.at(index);
    }
    inline          double& mean    (size_t index) {
        return _mean.at(index);
    }
    inline const    double& variance(size_t index) const {
        return _variance.at(index);
    }
    inline          double& variance(size_t index) {
        return _variance.at(index);
    }
    inline static double normalDistribution(double m, double v, double x, bool ignore_const = false) {
        return exp(-pow(x-m,2.0) / (ignore_const ? v : 2.0*v)) / sqrt(ignore_const ? v : v*M_PI_2);
    }
    inline static double logNormalDistribution(double m, double v, double x, bool ignore_const = false) {
        return -pow(x-m,2.0) / (ignore_const ? v : 2.0*v) - 0.5*log(ignore_const ? v : v*M_PI_2);
    }
    double* mean_data(){
        return _mean.data();
    }
    double* variance_data(){
        return _variance.data();
    }
};

std::vector<double> softmax(const std::vector<double>&);

template<typename To, typename From>
To dummy_cast(From ptr){
    return static_cast<To>(static_cast<void*>(ptr));
}

}