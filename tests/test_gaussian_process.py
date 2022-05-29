import dummyml as dmy
import numpy as np
import pytest
import os

def test_init():
    gp = dmy.gaussian_process(0.2)
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

def test_save_load():
    x = np.random.randn(100,3)
    y = np.random.randn(100)
    
    gp = dmy.gaussian_process(0.2, dmy.kernel.RadialBasisFunctionKernel)
    gp.fit(x,y)

    inputX = np.random.randn(3)
    results = gp(inputX)

    model_dir = "./model/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "gp.dmy"

    gp.save(model_path)

    gp2 = dmy.gaussian_process()
    gp2.load(model_path)

    assert np.all(results == gp2(inputX))

    gp3 = dmy.gaussian_process(model_path)
    assert np.all(results == gp3(inputX))

    with pytest.raises(Exception) as e:
        gp.load('./whateverFileDoesntExist.dmy')