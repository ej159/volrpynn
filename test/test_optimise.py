import numpy as np
import volrpynn as v

class TestModel():
    """A mock model"""
    def __init__(self):
        self.called_predict = 0
        self.called_backward = 0
    def predict(self, x, t):
        self.called_predict += 1
        return x
    def backward(self, error, backwards):
        self.called_backward += 1
        return error

def test_gradient_descent_train():
    model = TestModel()
    optimiser = v.GradientDescentOptimiser(0.1, decoder = lambda x: x)
    error = v.sum_squared_error
    xs = np.zeros((10, 2))
    ys = np.zeros((10, 1))
    optimiser.test(model, xs, ys)
    assert model.called_predict == 10
    assert model.called_backward == 0

def test_gradient_descent_test():
    model = TestModel()
    optimiser = v.GradientDescentOptimiser(0.1, decoder = lambda x: x)
    error = v.sum_squared_error
    xs = np.zeros((10, 2))
    ys = np.zeros((10, 1))
    optimiser.train(model, xs, ys)
    assert model.called_predict == 10
    assert model.called_backward == 10
