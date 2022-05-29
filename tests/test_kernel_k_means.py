import dummyml
import numpy as np
import pytest

def test_fit():
    input_dim = 784
    k = 10
    dataset_size = 60

    x = np.random.rand(dataset_size,input_dim)
    kkms = dummyml.kernel_k_means(k, x, dummyml.kernel.RadialBasisFunctionKernel)
    y = np.random.randint(0,k,size=dataset_size, dtype=np.int32)

    initial_inertia = kkms.inertia

    for i in range(10):
        kkms.fit(y)
    
    inertia_after_fit = kkms.inertia

    assert inertia_after_fit < initial_inertia

def test_large_dataset():
    input_dim = 3
    k = 3
    dataset_size = 2000
    x = np.random.rand(dataset_size,input_dim)
    kkms = dummyml.kernel_k_means(k, x, dummyml.kernel.RadialBasisFunctionKernel)
    y = np.random.randint(0,k,size=dataset_size, dtype=np.int32)

    initial_inertia = kkms.inertia

    kkms.fit(y)
    
    inertia_after_fit = kkms.inertia

    assert inertia_after_fit < initial_inertia

