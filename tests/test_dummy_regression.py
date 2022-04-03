import dummyml
import numpy as np

def test_inference():
    input_dim = 5
    drgsn = dummyml.dummy_regression(input_dim)
    result = drgsn(np.zeros(input_dim))
    assert result[0] == 1.0

    