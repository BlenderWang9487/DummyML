import dummyml
import numpy as np
import pytest

def test_fit_predict():
    input_dim = 784
    k = 10
    dataset_size = 60
    kms = dummyml.k_means(input_dim,k)

    x = np.random.rand(dataset_size,input_dim)
    y = np.random.randint(0,k,size=dataset_size, dtype=np.int32)

    for i in range(10):
        kms.fit(x,y)
    
    assert kms(np.random.rand(input_dim)) < k