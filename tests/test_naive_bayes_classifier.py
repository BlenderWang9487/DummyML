import dummyml
import numpy as np
import pytest
import os

def test_fit():
    class_num = 10
    input_dim = 28*28
    dataset_size = 6000
    nbclsfr = dummyml.naive_bayes_classifier(input_dim,class_num)
    nbclsfr.fit(np.zeros((dataset_size,input_dim)), np.ones((dataset_size,class_num)))
    
    result = nbclsfr(np.ones((input_dim,)))
    assert result.size == class_num
    sum_of_result = sum(result)
    assert sum_of_result == pytest.approx(1,0.01)

def test_save_load():
    class_num = 10
    input_dim = 28*28
    dataset_size = 600
    nbclsfr = dummyml.naive_bayes_classifier(input_dim,class_num)
    nbclsfr.fit(np.zeros((dataset_size,input_dim)), np.ones((dataset_size,class_num)))

    model_path = "./model/test.dmy"
    nbclsfr.save(model_path)
    assert os.path.isfile(model_path)

    nbclsfr_2 = dummyml.naive_bayes_classifier(1,1)
    nbclsfr_2.load(model_path)
    assert nbclsfr_2(np.ones((input_dim,))).size == class_num

    with pytest.raises(Exception) as e:
        nbclsfr_2.load('./whateverFileDoesntExist.dmy')
