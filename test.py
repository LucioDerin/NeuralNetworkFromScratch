# test of all the classes

import numpy as np
from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data
from DiyKeras.DenseLayer import DenseLayer
from DiyKeras.ActivationFunctions import ActivationReLU, ActivationSoftmax
from DiyKeras.LossFunctions import CategoricalCrossEntropy, Accuracy

if __name__ == "__main__":
    # Import dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = DenseLayer(2, 3)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = ActivationReLU()
    # Create second Dense layer with 3 input features (dimensionality of the output
    # of the last layer) and 3 output values
    dense2 = DenseLayer(3, 3)
    # Create Softmax activation (to be used with Dense layer):
    activation2 = ActivationSoftmax()

    # Forward Passes:

    # First Layer:
    # Make a forward pass of our training data through this layer
    dense1.forwardPass(X)
    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forwardPass(dense1.output)

    # Second Layer:
    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forwardPass(activation1.output)
    # Make a forward pass through activation function
    # it takes the output of second dense layer here
    activation2.forwardPass(dense2.output)
    # Confidence scores should be about 0.33 for each class
    # Since weights are randomly initialized
    print(activation2.output[:5])

    # Loss function:
    # Perform a forward pass through loss function
    # it takes the output of second dense layer here and returns loss
    lossFunction = CategoricalCrossEntropy()
    accuracyFunction = Accuracy()
    loss = lossFunction.calculate(y,activation2.output)
    acc = accuracyFunction.calculate(y,activation2.output)

    # Print loss value
    print('loss:', loss)
    print('acc:', acc)
    
    plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
    plt.title("Training Set")
    plt.show()