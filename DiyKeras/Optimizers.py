import numpy as np

class GradientDescend:

    # Constructor
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        '''
        Updates the parameters of the layer based on the gradient.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases