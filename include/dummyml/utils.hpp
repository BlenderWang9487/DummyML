#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

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

class kernel{
public:
    kernel() = default;
    virtual double operator()(const double &x1, const double &x2) = 0;
    virtual double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) = 0;
};

class linear_kernel: public kernel{
public:
    linear_kernel() = default;
    // test unique ptr destruction
    ~linear_kernel(){
        std::cout<<"linear_kernel dtr:"<<std::endl;
    }
    double operator()(const double &x1, const double &x2){
        return x1*x2;
    }
    double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2){
        return x1.dot(x2);
    }
};

class RBF_kernel: public kernel{
private:
    double gamma;
public:
    RBF_kernel(double g = 0.1): kernel(), gamma(g){}
    void set_gamma(double g){
        gamma = g;
    }
    double operator()(const double &x1, const double &x2){
        return exp(-(x1-x2)*(x1-x2)*gamma);
    }
    double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2){
        auto dis = x1-x2;
        return exp(-dis.dot(dis)*gamma);
    }
};

std::vector<double> softmax(const std::vector<double>&);

template<typename To, typename From>
To dummy_cast(From ptr){
    return static_cast<To>(static_cast<void*>(ptr));
}

}