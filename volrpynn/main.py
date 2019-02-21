import json
import sys
import numpy as np
import volrpynn as v

class Main():
    """A runtime class that accepts a model and exposes a 'train' method
       to train that model with a given optimiser, given data via std in"""

    def __init__(self, model, parameters=None,
            translation=v.LinearTranslation()):
        self.model = model
        if isinstance(parameters, str):
            self._load_parameters_file(parameters)
        if isinstance(parameters, np.ndarray):
            self._load_parameters(parameters)
        if not isinstance(translation, v.Translation):
            raise ValueError('Translator must be a Translation')
        self.translation = translation

    def _load_parameters_file(self, file_name):
        parameters = np.load(file_name)
        self._load_parameters(parameters)

    def _load_parameters(self, parameters):
        for index in range(len(self.model.layers) - 1):
            layer = self.model.layers[index]
            weights, biases = parameters[:2]
            layer.biases = biases
            layer.set_weights(weights)
            # Continue with next element in tuple
            parameters = parameters[2:]

    def _load_data(self):
        if len(sys.argv) < 3:
            raise Exception("Training input and training labels expected as "+\
                            "either argument data or filenames")
        xs_text, ys_text = (sys.argv[1], sys.argv[2])
        if type(xs_text) == str and type(ys_text) == str:
            return (self._load_file(xs_text), self._load_file(ys_text))
        else:
            return xs_text, ys_text

    def _load_file(self, filename):
        with open(filename, 'r') as fp:
            return fp.read()

    def train(self, optimiser, xs=None, ys=None, split=0.8):
        """Trains and tests the model loaded in this class with the given
        optimiser, input data, expected output data and testing/training
        split

        Args:
        optimiser -- The optimisation algorithm that trains the model
        xs -- The input data, will later be normalised
        ys -- Expected categorical output labels
        split -- Testing/training split. Defaults to 0.8 (80%)

        Returns:
        A Report of the training and testing run
        """
        if not isinstance(xs, np.ndarray) or not isinstance(ys, np.ndarray):
            xs, ys = self._load_data()
            xs = np.array(json.loads(xs))
            ys = np.array(json.loads(ys))

        # Normalise data
        xs = self.translation.to_current(xs)

        # Split training/testing
        split = int(len(xs) * split)
        x_train = xs[:split]
        y_train = ys[:split]
        x_test = xs[split:]
        y_test = ys[split:]
        assert len(x_train) > 0 and len(x_test) > 0, "Must have at least 5 data points"
        _, errors, _ = optimiser.train(self.model, x_train, y_train, v.SoftmaxCrossEntropy())
        report = optimiser.test(self.model, x_test, y_test, v.ErrorCostCategorical())

        reportDict = report.toDict()
        reportDict['train_errors'] = errors # Include training errors

        # Add network weights and biases
        parameters = []
        for layer in self.model.layers[:-1]: # Exclude decode layer
            parameters.append(layer.get_weights().tolist())
            parameters.append(layer.biases.tolist())
        reportDict['parameters'] = parameters

        # Return a JSON version of the report to stdout
        print(json.dumps(reportDict))
