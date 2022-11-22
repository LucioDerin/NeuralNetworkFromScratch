import numpy as np

class VanillaGradientDescent:

    # Constructor
    def __init__(self, learningRate=1.0):
        self.learningRate = learningRate

    def updateParams(self, layer):
        '''
        Updates the parameters of the layer based on the gradient.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''

        layer.weights -= self.learningRate * layer.dweights
        layer.biases -= self.learningRate * layer.dbiases

class DecayGradientDescent(VanillaGradientDescent):

    # Constructor
    def __init__(self, decay = 0., learningRate=1.0):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
    
    # Call once before any parameter updates
    def calcCurrentLearningRate(self):
        '''
        Calculates the new learning rate with decay.
        Modifies:
        @ self.currentLearningRate
        '''
        if self.decay != 0.:
            self.currentLearningRate = self.learningRate * ((1. + self.decay * self.iterations)**-1)

    def updateParams(self, layer):
        '''
        Updates the parameters of the layer based on the gradient and decaying learning rate.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''
        # update learning rate
        if self.decay != 0:
            self.calcCurrentLearningRate()

        layer.weights -= self.currentLearningRate * layer.dweights
        layer.biases -= self.currentLearningRate * layer.dbiases
        # update iterations' counter
        self.iterations += 1

class NesterovGradientdescent(DecayGradientDescent):

    # Constructor
    def __init__(self, decay = 0., momentum = .5, learningRate=1.0):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def updateParams(self, layer):

        # update learning rate
        if self.decay != 0:
            self.calcCurrentLearningRate()

        # If layer does not contain momentum arrays, create them
        # filled with zeros
        if not hasattr(layer, 'weightMomentums'):
            layer.weightMomentums = np.zeros_like(layer.weights)
        
            # If there is no momentum array for weights
            # The array doesn't exist for biases yet either.
            layer.biasMomentums = np.zeros_like(layer.biases)
        
        # Build weight updates with momentum - take previous
        # updates multiplied by retain factor and update with current gradients
        weightUpdates = self.momentum * layer.weightMomentums - self.currentLearningRate * layer.dweights
        layer.weightMomentums = weightUpdates
        # Build bias updates
        biasUpdates = self.momentum * layer.biasMomentums - self.currentLearningRate * layer.dbiases
        layer.biasMomentums = biasUpdates

        layer.weights += weightUpdates
        layer.biases += biasUpdates

        # update iterations' counter
        self.iterations += 1