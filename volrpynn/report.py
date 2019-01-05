import numpy as np

class Report():
    """A report that stored and renders information from testing runs"""
    pass

class ErrorCost(Report):

    def __init__(self, cost_function):
        self.costs = []
        self.hits = []

        assert callable(cost_function)
        self.cost_function = cost_function

    def add(self, output, expected):
        self.hits.append(np.allclose(output, expected))
        self.costs.append(self.cost_function(output, expected))

    def __repr__(self):
        hits = np.array(self.hits)
        acc = hits[hits == True].sum() / len(hits)
        return "Report: {} tests with an accuracy of {}".format(len(hits), acc)

class ErrorCostCategorical(ErrorCost):
    
    def add(self, output, expected):
        result = np.zeros(len(output))
        result[np.argmax(output)] = 1
        self.hits.append(np.allclose(result, expected))
        self.costs.append(self.cost_function(result, expected))

