import numpy as np
import volrpynn as v

class Report():
    """A report that stored and renders information from testing runs"""
    pass

class ErrorCost(Report):

    def __init__(self, error_function = v.SumSquared()):
        self.errors = []
        self.hits = []

        assert callable(error_function)
        self.error_function = error_function

    def add(self, output, expected):
        self.hits.append(np.allclose(output, expected))
        self.errors.append(self.error_function(output, expected))

    def accuracy(self):
        hits = np.array(self.hits)
        return hits[hits == True].sum() / len(hits)

    def __repr__(self):
        hits = np.array(self.hits)
        return "Report: {} tests with accuracy {} and error {}".format(len(hits), self.accuracy(), sum(self.errors))

class ErrorCostCategorical(ErrorCost):

    def __init__(self, error_function = v.SumSquared()):
        super(ErrorCostCategorical, self).__init__(error_function)
        self.outputs = []

    def add(self, output, expected):
        self.outputs.append(output)
        actual = np.zeros(len(output))
        actual[np.argmax(output)] = 1
        self.hits.append(np.allclose(actual, expected))

        errors = self.error_function(output, expected)
        self.errors.append(errors)

