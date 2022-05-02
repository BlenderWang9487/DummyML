import dummyml
import numpy as np
import pytest

def test_fit_predict():
    input_dim = 784
    k = 10
    dataset_size = 60
    kms = dummyml.k_means(input_dim,k,np.random.rand(k*input_dim))

    x = np.random.rand(dataset_size,input_dim)
    y = np.zeros((dataset_size,1),dtype=np.float64)

    for i in range(10):
        kms.fit(x,y)
    
    assert kms(np.random.rand(input_dim)).size == 1