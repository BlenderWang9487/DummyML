#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <iostream>
#include <cmath>

namespace py = pybind11;

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
    enum type{
        LinearKernel = 0,
        RadialBasisFunctionKernel
    };
    kernel() = default;
    virtual double operator()(const double &x1, const double &x2) = 0;
    virtual double operator()(
        Eigen::Ref<Eigen::VectorXd,0,Eigen::InnerStride<>> x1,
        Eigen::Ref<Eigen::VectorXd,0,Eigen::InnerStride<>> x2) = 0;
};

class linear_kernel: public kernel{
public:
    linear_kernel() = default;
    double operator()(const double &x1, const double &x2){
        return x1*x2;
    }
    double operator()(
        Eigen::Ref<Eigen::VectorXd,0,Eigen::InnerStride<>> x1,
        Eigen::Ref<Eigen::VectorXd,0,Eigen::InnerStride<>> x2){
        return x1.dot(x2);
    }
};

class radial_basis_function_kernel: public kernel{
private:
    double gamma;
public:
    radial_basis_function_kernel(double g = 0.1): kernel(), gamma(g){}
    void set_gamma(double g){
        gamma = g;
    }
    double operator()(const double &x1, const double &x2){
        return exp(-gamma*(x1-x2)*(x1-x2));
    }
    double operator()(
        Eigen::Ref<Eigen::VectorXd,0,Eigen::InnerStride<>> x1,
        Eigen::Ref<Eigen::VectorXd,0,Eigen::InnerStride<>> x2){
        Eigen::VectorXd dis = x1-x2;
        return exp(-gamma*dis.dot(dis));
    }
};

std::unique_ptr<kernel> get_kernel(kernel::type);

std::vector<double> softmax(const std::vector<double>&);

template<typename To, typename From>
To dummy_cast(From ptr){
    return static_cast<To>(static_cast<void*>(ptr));
}

} // namespace dummyml

void export_utils(py::module_ &m);