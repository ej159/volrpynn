import abc
import numpy as np
import volrpynn as v

class Report():
    """A report that stored and renders information from testing runs"""

    @abc.abstractmethod
    def toDict(self):
        """Converts the report to a simple dictionary, useful for
        serialisation."""
        pass

class ErrorCost(Report):
    """A report based on an error function"""

    def __init__(self, error_function=v.SumSquared()):
        self.errors = []
        self.hits = []

        assert callable(error_function)
        self.error_function = error_function

    def add(self, output, expected):
        """Adds the output to the error report"""
        self.hits.append(np.allclose(output, expected))
        self.errors.append(self.error_function(output, expected).tolist())

    def accuracy(self):
        """Retrieves the accuracy of all the trials in the report"""
        hits = np.array(self.hits)
        return hits[hits].sum() / len(hits)

    def toDict(self):
        return {'errors': self.errors, 'hits': self.hits,
                'accuracy': self.accuracy()}

    def __repr__(self):
        hits = np.array(self.hits)
        return "Report: {} tests with accuracy {} and error {}"\
                 .format(len(hits), self.accuracy(), sum(self.errors))

class ErrorCostCategorical(ErrorCost):
    """Report based on categorical data"""

    def __init__(self, error_function=v.SumSquared()):
        super(ErrorCostCategorical, self).__init__(error_function)
        self.outputs = []

    def add(self, output, expected):
        self.outputs.append(output)
        actual = np.zeros(len(output))
        actual[np.argmax(output)] = 1
        self.hits.append(np.allclose(actual, expected))

        errors = self.error_function(output, expected).tolist()
        self.errors.append(errors)

    def toDict(self):
        sup = ErrorCost.toDict(self)
        sup['outputs'] = list(map(lambda x: x.tolist(), self.outputs))
        return sup
