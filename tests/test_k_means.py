import dummyml
import numpy as np
import pytest
import os

def test_fit_predict():
    input_dim = 784
    k = 10
    dataset_size = 60
    kms = dummyml.k_means(input_dim,k)

    x = np.random.rand(dataset_size,input_dim)
    y = np.random.randint(0,k,size=dataset_size, dtype=np.int32)

    initial_iner = kms.inertia

    for i in range(10):
        kms.fit(x,y)

    after_fit_iner = kms.inertia
    
    assert kms(np.random.rand(input_dim)) < k
    assert after_fit_iner < initial_iner

def test_save_load():
    input_dim = 784
    k = 10
    dataset_size = 60
    kms = dummyml.k_means(input_dim,k)

    x = np.random.rand(dataset_size,input_dim)
    y = np.random.randint(0,k,size=dataset_size, dtype=np.int32)

    for i in range(10):
        kms.fit(x,y)
    
    means_before_save = kms.means()

    model_dir = "./model/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "kms.dmy"

    kms.save(model_path)

    kms2 = dummyml.k_means(input_dim,k)
    kms2.load(model_path)

    means_after_save = kms2.means()

    assert np.all(means_after_save == means_before_save)

    with pytest.raises(Exception) as e:
        kms.load('./whateverFileDoesntExist.dmy')