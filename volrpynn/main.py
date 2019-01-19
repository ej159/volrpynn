import volrpynn as v
import numpy as np
import json
import sys

class Main():
    """A runtime class that accepts a model and exposes a 'train' method
       to train that model with a given optimiser, given data via std in"""

    def __init__(self, model):
        self.model = model

    def train(self, optimiser):
        if len(sys.argv) < 3:
            raise Exception("Training input and training labels expected via std in")
        (xs_text, ys_text) = sys.argv[1:]
        xs = np.array(json.loads(xs_text))
        ys = np.array(json.loads(ys_text))
        split = int(len(xs) * 0.8) # 80% training data
        x_train = xs[:split]
        y_train = ys[:split]
        x_test = xs[split:]
        y_test = ys[split:]
        assert len(x_train) > 0 and len(x_test) > 0, "Must have at least 5 data points"
        optimiser.train(self.model, x_train, y_train, v.SumSquared())
        report = optimiser.test(self.model, x_test, y_test, v.ErrorCostCategorical())
        print(report)
