import  numpy as np
from DiyKeras.LossFunctions import CategoricalCrossEntropy

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

    def backwardPass(self, dvalues):
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

    def backwardPass(self, dvalues):
        '''
        Evaluates the backward pass with current weights an biases on a batch of data. Results are
        stored in the public member self.dinputs.
        Parameters:
        @dvalues: gradient of the previous layer (which is the loss function) wrt inputs, 
        shape must be (nBatch,nCategories);
        Modifies:
        @self.dinputs: array of shape (nBatch,nCategories), gradient of the activation function;
        '''
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # iterating trough outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - single_output@single_output.T
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = jacobian_matrix@single_dvalues

# Softmax activation + Categorical Cross Entropy loss
class ActivationSoftmaxCategoricalCrossEntropy():
    
    # Constructor
    def __init__(self):
        '''
        Initializer, instantiates the cross-entropy loss and the softmax activation.
        '''
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossEntropy()

    def forwardPass(self, yTrue, inputs):
        '''
        Applies the softmax activation function and calculates the CE loss.
        Parameters:
        @yTrue: array of shape (nBatch,) if belonging category is represented with its index or (nBatch,nCategories) if one-hot, ground truth labels;
        @inputs: batch of linear combination of the inputs of the layer.
        Returns:
        @loss: float, cross-entropy loss of the batch of predictions;
        '''
        # Output layer's activation function
        self.activation.forwardPass(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(yTrue, self.output)

    def backwardPass(self, yTrue, yPred):
        '''
        Evaluates the backward pass with current weights an biases on a batch of data. Results are
        stored in the public member self.dinputs.
        Parameters:
        @yTrue: array of shape (nBatch,) if belonging category is represented with its index or (nBatch,nCategories) if one-hot, ground truth labels;
        @yPred: array of shape (nBatch, nCategories), confidence scores (SoftMax's output).
        Modifies:
        @self.dinputs: array of shape (nBatch,nCategories), gradient of the softmax+cross-entropy combination;
        '''
        # Number of samples
        samples = len(yPred) if isinstance(yPred,list) else yPred.shape[0]
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1)
        # Copy so we can safely modify
        self.dinputs = yPred.copy()
        # Calculate gradient 
        # by subtracting 1 only to the yPred's component corresponding to the belonging category
        self.dinputs[range(samples), yTrue] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples