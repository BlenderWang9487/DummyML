import dummyml
import numpy as np

def test_init():
    dctr = dummyml.dummy_clustering()
    assert dctr.k == 2

    dctr = dummyml.dummy_clustering(5)
    assert dctr.k == 5

    dctr = dummyml.dummy_clustering(k=10)
    assert dctr.k == 10

def test_fit():
    y = np.array([100,100,100], dtype=np.int32)
    dctr = dummyml.dummy_clustering(k=10)

    dctr.fit(y)

    assert y[0] != 100
    assert y[0] < 10 and y[0] >= 0

def test_pred():
    x = np.random.randn(100)
    dctr = dummyml.dummy_clustering(k=5)

    pred = dctr(x)
    assert pred < 5 and pred >= 0

    dctr = dummyml.dummy_clustering(k=1)

    pred = dctr(x)
    assert pred == 0


