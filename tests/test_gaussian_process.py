import dummyml as dmy
import numpy as np

def test_init():
    gp = dmy.gaussian_process(0.2, dmy.kernel.LinearKernel)
    assert gp.run_kernel(4,4) == 16.0

def test_fit():
    x = np.random.randn(100,3)
    y = np.random.randn(100)
    
    gp = dmy.gaussian_process(0.2, dmy.kernel.RadialBasisFunctionKernel)
    gp.fit(x,y)

    assert gp.run_kernel(4,4) == 1.0

def test_predict():
    x = np.random.randn(100,3)
    y = np.random.randn(100)
    
    gp = dmy.gaussian_process(x, y, 0.2, dmy.kernel.RadialBasisFunctionKernel)
    
    assert len(gp(x[0])) == 2