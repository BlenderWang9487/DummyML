from dummyml import dummyml
import numpy as np

def test_inference():
    dclsfr = dummyml.dummy_classifier(5,2)
    result = dclsfr(np.zeros(5))
    assert result[0] == 1.0