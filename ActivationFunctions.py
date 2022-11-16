import  numpy as np

# ReLU activation function class
class ActivationReLU:

    def forwardPass(self, inputs):
        '''
        Applies the ReLU activation function on the inputs. Results are stored in public member self.output.
        Parameters:
        @inputs: batch of input points.
        '''
        # max between zero and each value of inputs
        self.output = np.maximum(0, inputs)

# Softmax activation
class ActivationSoftmax:

    def forwardPass(self, inputs):
        '''
        Applies the softmax activation function on the inputs. Results are stored in public member self.output.
        Parameters:
        @inputs: batch of input points.
        '''
        # exponentiate the shifted data so that the argument of exp is at most 0
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the output to get the probability interpretation
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities