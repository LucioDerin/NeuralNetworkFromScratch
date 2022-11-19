import numpy as np
from DiyKeras.ActivationFunctions import ActivationReLU, ActivationSoftmax

class DenseLayer:
    # the scale factor of the gaussian pdf by which weights are initialized
    scale = 0.01
    # the offset to the weights initialization
    epsilon = 0

    # Constructor
    def __init__(self,dInput,nNeurons):
        '''
        Constructor, initializes the weights and the biases of the layer.
        Parameters:
        @dInput: dimensionality of the input space (with respect to to this layer);
        @nNeurons: number of neurons in this layer, i.e. layer's width;
        Modifies:
        @self.weights: weights of the layer;
        @self.biases: biases of the layer;
        '''
        self.weights = self.scale * np.random.randn(dInput,nNeurons) + self.epsilon
        self.biases = np.zeros((1,nNeurons))

    def forwardPass(self,inputs):
        '''
        Evaluates the forward pass with current weights an biases on a batch of data. Results are
        stored in the public member self.output.
        Parameters:
        @inputs: batch of data to forward pass, shape must be (nBatch,dInput)
        Modifies:
        @self.output: array of shape (nBatch,nNeurons), result of the forward pass;
        '''
        # storing inputs for backpropagation
        self.inputs = inputs
        self.output = inputs@self.weights + self.biases

    def backwardPass(self, dvalues):
        '''
        Evaluates the backward pass with current weights and biases on a batch of data. Results are
        stored in the public member self.dweights, self.dbiases, self.dinputs.
        Parameters:
        @dvalues: gradient of the activation function, shape must be (nBatch,nNeurons)
        Modifies:
        @self.dweights: array of shape (dInputs, nNeurons), gradient with respect to the weights; 
        @self.dbiases: array of shape (1,nNeurons), gradient with respect to the biases; 
        @self.dinputs: array of shape (nBatch,dInputs), gradient with respect to the inputs (needed to chain the next layer); 
        '''
        # Gradient wrt weights
        self.dweights = self.inputs.T@dvalues
        # Gradient wrt biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient wrt inputs
        self.dinputs = dvalues@self.weights.T