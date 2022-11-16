#*****************************#
#Class DenseLayer:
#Inheritance: None
#Author: Derin Lucio
#*****************************#

#*******DESCRIPTION:*********#
# Class that implements a layer of fully connected (with respect to the previous layer) neurons,
# with free choice of the number of inputs and neuron in the layer.

#*******CONSTRUCTORS:********#
# __init__(self,nInputs,nNeurons)

#**********SETTERS:**********#


#**********GETTERS:**********#


#********OPERATORS:**********#


#**********METHODS:**********#
# forwardPass(self,inputs): implements the forward pass of the inputs through the layer;

#********FUNCTIONS:*********#

#****************************#

import numpy as np
from ActivationFunctions import ActivationReLU, ActivationSoftmax

class DenseLayer:
    # the scale factor of the gaussian pdf by which weights are initialized
    scale = 0.01
    # the offset to the weights initialization
    epsilon = 0
    def __init__(self,dInput,nNeurons):
        '''
        Constructor, initializes the weights and the biases of the layer.
        Parameters:
        @dInput: dimensionality of the input space (with respect to to this layer);
        @nNeurons: number of neurons in this layer, i.e. layer's width;
        '''
        self.weights = self.scale * np.random.randn(dInput,nNeurons) + self.epsilon
        self.biases = np.zeros((1,nNeurons))

    def forwardPass(self,input):
        '''
        Evaluates the forward pass with current weights an biases on a batch of data. Results are
        stored in the public member self.output.
        Parameters:
        @input: batch of data to forward pass, shape must be (nBatch,dInput)
        '''
        self.output = input@self.weights + self.biases

if __name__ == "__main__":
    from nnfs.datasets import spiral_data

    ## Create dataset
    #X, y = spiral_data(samples=100, classes=3)
    ## Create Dense layer with 2 input features and 3 output values
    #dense1 = DenseLayer(2, 3)
    ## Perform a forward pass of our training data through this layer
    #dense1.forwardPass(X)
    ## Let's see output of the first few samples:
    #print(dense1.output[:5])

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