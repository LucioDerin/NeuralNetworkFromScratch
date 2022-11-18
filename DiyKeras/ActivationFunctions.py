import  numpy as np

# ReLU activation function class
class ActivationReLU:

    def forwardPass(self, inputs):
        '''
        Applies the ReLU activation function. Results are stored in public member self.output.
        Parameters:
        @inputs: batch of linear combination of the inputs of the layer.
        Modifies:
        @self.output: array of shape (nBatch,nNeurons), non linear output of the layer;
        '''
        # store inputs for backpropagation
        self.inputs = inputs
        # max between zero and each value of inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        '''
        Evaluates the backward pass with current weights an biases on a batch of data. Results are
        stored in the public member self.dinputs.
        Parameters:
        @dvalues: gradient of the previous layer wrt inputs, shape must be (nBatch,nNeurons);
        Modifies:
        @self.dinputs: array of shape (nBatch,nNeurons), gradient of the activation function;
        '''
        # copy the previous layer gradient to apply the cuts where dvalues<0
        self.dinputs = dvalues.copy()
        # Gradient wrt activation function's inputs
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation
class ActivationSoftmax:

    def forwardPass(self, inputs):
        '''
        Applies the softmax activation function. Results are stored in public member self.output.
        Parameters:
        @inputs: batch of linear combination of the inputs of the layer.
        Modifies:
        @self.output: array of shape (nBatch,nNeurons), non linear output of the layer;
        '''
        # exponentiate the shifted data so that the argument of exp is at most 0
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the output to get the probability interpretation
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities