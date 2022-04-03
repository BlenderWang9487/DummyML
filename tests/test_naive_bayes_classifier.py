import dummyml
import numpy as np
import pytest

def test_fitting():
    class_num = 10
    input_dim = 28*28
    dataset_size = 60000
    nbclsfr = dummyml.naive_bayes_classifier(input_dim,class_num)
    nbclsfr.fit(np.zeros((dataset_size,input_dim)), np.ones((dataset_size,class_num)))
    
    result = nbclsfr(np.ones((input_dim,)))
    assert result.size == class_num
    sum_of_result = sum(result)
    assert sum_of_result == pytest.approx(1,0.01)
    