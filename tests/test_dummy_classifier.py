from dummyml import dummyml
import numpy as np

def test_inference():
    dim = 10000000
    dclsfr = dummyml.dummy_classifier(5,dim)
    result = dclsfr(np.zeros(5))
    assert result[0] == 1.0
    flag = True
    for r in result[1:]:
        if r != 0.0:
            flag = False
            break
    assert flag