#include <utils.hpp>

namespace dummyml{

std::vector<double> softmax(const std::vector<double>& v){
    std::vector<double> ret(v.size());
    double sum = 0.0;
    for(size_t i = 0;i < v.size();++i)
        sum += ret[i] = exp(v[i]);
    for(size_t i = 0;i < v.size();++i)
        ret[i] /= sum;
    return ret;
}

std::unique_ptr<kernel> get_kernel(kernel::type k_type){
    switch (k_type)
    {
    case kernel::type::LinearKernel:
        return std::make_unique<linear_kernel>();
    case kernel::type::RadialBasisFunctionKernel:
        return std::make_unique<radial_basis_function_kernel>();
    default:
        return std::make_unique<linear_kernel>();
    }
}

} // namespace dummyml

void export_utils(py::module_ &m){
    py::class_<dummyml::kernel> k(m, "kernel");
    py::enum_<dummyml::kernel::type>(k, "type")
        .value("LinearKernel", dummyml::kernel::type::LinearKernel)
        .value("RadialBasisFunctionKernel", dummyml::kernel::type::RadialBasisFunctionKernel)
        .export_values();
        
}
