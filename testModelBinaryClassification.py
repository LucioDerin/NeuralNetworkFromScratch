from nnfs.datasets import spiral_data
import numpy as np

from DiyKeras.Model import Model
from DiyKeras.Layers import DenseLayer
from DiyKeras.ActivationFunctions4Model import ReLU, Sigmoid
from DiyKeras.LossFunctions4Model import BinaryCrossentropy
from DiyKeras.Optimizers import Adam
from DiyKeras.Accuracy4Model import AccuracyCategorical

if __name__=="__main__":
    # Create train and test dataset
    X, y = spiral_data(samples=100, classes=2)
    X_test, y_test = spiral_data(samples=100, classes=2)
    # Reshape labels to be a list of lists
    # Inner list contains one output (either 0 or 1)
    # per each output neuron, 1 in this case
    y = y.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # Instantiate the model
    model = Model()
    # Add layers
    model.add(DenseLayer(2, 64, lambda_2w=5e-4, lambda_2b=5e-4))
    model.add(ReLU())
    model.add(DenseLayer(64, 1))
    model.add(Sigmoid())
    # Set loss, optimizer and accuracy objects
    model.set(
        loss=BinaryCrossentropy(),
        optimizer=Adam(decay=5e-7),
        accuracy=AccuracyCategorical(binary=True)
    )
    # Finalize the model
    model.finalize()
    # Train the model
    model.train(X, y, validationData=(X_test, y_test), epochs=10000, printEvery=100)