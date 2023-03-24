import numpy as np

# Common accuracy class
class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, y, predictions):
        # Get comparison results -> "virtual" method
        comparisons = self.compare(y, predictions)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        # Return accuracy
        return accuracy

# Accuracy calculation for regression model
class AccuracyRegression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None
    # Calculates precision value
    # based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    # Compares predictions to the ground truth values
    def compare(self, y, predictions):
        return np.absolute(predictions - y) < self.precision

# Accuracy calculation for classification model
class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
    # No initialization is needed
    def init(self, y):
        pass
    # Compares predictions to the ground truth values
    def compare(self, y, predictions):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y