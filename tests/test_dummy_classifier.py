import dummyml
import numpy as np

def test_inference():
    output_dim = 10000
    input_dim = 5
    dclsfr = dummyml.dummy_classifier(input_dim,output_dim)
    result = dclsfr(np.zeros(input_dim))
    assert result[0] == 1.0
    flag = True
    for r in result[1:]:
        if r != 0.0:
            flag = False
            break
    assert flag

def test_wrong_class_num():
    output_dim = 0
    input_dim = 5
    dclsfr = dummyml.dummy_classifier(input_dim,output_dim)
    result = dclsfr(np.zeros(input_dim))
    assert result.shape == (1,)
    