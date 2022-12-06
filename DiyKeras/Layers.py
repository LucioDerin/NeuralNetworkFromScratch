import numpy as np
#from DiyKeras.ActivationFunctions import ReLU, Softmax

class DenseLayer:
    # the scale factor of the gaussian pdf by which weights are initialized
    scale = 0.01
    # the offset to the weights initialization
    epsilon = 0

    # Constructor
    def __init__(self, dInputs, nNeurons, lambda_1w=0, lambda_2w=0, lambda_1b=0, lambda_2b=0):
        '''
        Constructor, initializes the weights and the biases of the layer.
        Parameters:
        @dInput: dimensionality of the input space (with respect to to this layer);
        @nNeurons: number of neurons in this layer, i.e. layer's width;
        @lambda_1w: regularization parameter for the L1 norm of the weights;
        @lambda_2w: regularization parameter for the L2 norm of the weights;
        @lambda_1b: regularization parameter for the L1 norm of the biases;
        @lambda_2b: regularization parameter for the L2 norm of the biases;
        Modifies:
        @self.weights: weights of the layer;
        @self.biases: biases of the layer;
        '''
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(dInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
        # Set regularization strength
        self.lambda_1w = lambda_1w
        self.lambda_2w = lambda_2w
        self.lambda_1b = lambda_1b
        self.lambda_2b = lambda_2b

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
        If regularization parameters are non zero, it applies the regularization gradient.
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

        # Gradients on regularization

        # L1 on weights
        if self.lambda_1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.lambda_1w*dL1
        # L2 on weights
        if self.lambda_2w > 0:
            self.dweights += 2*self.lambda_2w*self.weights

        # L1 on biases
        if self.lambda_1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.lambda_1b*dL1
        # L2 on biases
        if self.lambda_2b > 0:
            self.dbiases += 2*self.lambda_2b*self.biases

        # Gradient wrt inputs
        self.dinputs = dvalues@self.weights.T

class Dropout:

    # Constructor
    def __init__(self, dr):
        '''
        Constructor, saves the dropout ratio.
        Parameters:
        @dr: dropout ratio defined as Ndrop/Ntot;
        '''
        # convert the dropout ratio to the probability of keeping a neuron for code simplicity
        self.p1 = 1-dr
    
    def forwardPass(self, inputs):
        '''
        Applies the binary mask on the forward pass, suppressing a fraction dr of the 
        previous layer's neurons' outputs.
        Parameters:
        @inputs: matrix of outputs of the previous layer;
        Modifies:
        @self.binaryMask: the (scaled!) dropout mask, gets stored for the backward pass;
        @self.output: masked output to be fed to the next layer;
        '''
        # Generate and save scaled mask
        self.binaryMask = np.random.binomial(1, self.p1, size=inputs.shape)/self.p1
        # Apply mask to output values
        self.output = inputs*self.binaryMask

    def backwardPass(self, dvalues):
        '''
        Calculates the backward pass of the dropout layer by multiplying the current gradient by the scaled
        binary mask.
        Parameters:
        @dvalues: gradient of the previous layer;
        Modifies:
        @self.dinputs: updated gradient;
        '''
        # Gradient on values
        self.dinputs = dvalues * self.binaryMask