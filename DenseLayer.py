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