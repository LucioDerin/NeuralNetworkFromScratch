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
        # Update iterations' counter
        self.iterations += 1

class MomentumGradientDescent(DecayGradientDescent):

    # Constructor
    def __init__(self, decay = 0., momentum = .5, learningRate=1.0):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def updateParams(self, layer):
        '''
        Updates the parameters of the layer based on the gradient, decaying learning rate and momentum.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''
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
        
        # Build weight updates with momentum: take previous
        # updates multiplied by retain factor and update with current gradients
        weightUpdates = self.momentum * layer.weightMomentums - self.currentLearningRate * layer.dweights
        layer.weightMomentums = weightUpdates
        # Build bias updates
        biasUpdates = self.momentum * layer.biasMomentums - self.currentLearningRate * layer.dbiases
        layer.biasMomentums = biasUpdates

        # Update the parameters
        layer.weights += weightUpdates
        layer.biases += biasUpdates

        # Update iterations' counter
        self.iterations += 1

class AdaGrad(DecayGradientDescent):

    # Constructor
    def __init__(self, learningRate=1., decay=0., epsilon=1e-7):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Update parameters
    def updateParams(self, layer):
        '''
        Updates the parameters of the layer based on the gradient with decaying and per-parameter tuned learning rate.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''

        # update learning rate
        if self.decay != 0:
            self.calcCurrentLearningRate()
        
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weightCache += layer.dweights**2
        layer.biasCache += layer.dbiases**2

        # Update the parameters
        layer.weights -= self.currentLearningRate * layer.dweights/(np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases -= self.currentLearningRate *layer.dbiases/(np.sqrt(layer.biasCache) + self.epsilon)
        
        # Update iterations' counter
        self.iterations += 1

class RMSProp(DecayGradientDescent):

    # Constructor
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7,rho=0.9):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Update parameters
    def updateParams(self, layer):
        '''
        Updates the parameters of the layer based on the gradient with decaying and per-parameter tuned learning rate and momentum on the cache.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''

        # update learning rate
        if self.decay != 0:
            self.calcCurrentLearningRate()
        
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCache = np.zeros_like(layer.biases)
        
        # Update cache with moving average
        layer.weightCache = self.rho * layer.weightCache + (1 - self.rho) * layer.dweights**2
        layer.biasCache = self.rho * layer.biasCache + (1 - self.rho) * layer.dbiases**2

        # Update the parameters
        layer.weights -= self.currentLearningRate * layer.dweights/(np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases -= self.currentLearningRate *layer.dbiases/(np.sqrt(layer.biasCache) + self.epsilon)
        
        # Update iterations' counter
        self.iterations += 1

class Adam(DecayGradientDescent):

    # Constructor
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # Update parameters
    def updateParams(self, layer):
        '''
        Updates the parameters of the layer based on the gradient with decaying and per-parameter tuned learning rate and momentum on the cache and gradient.
        Parameters:
        @layer: instance of DenseLayer class, layer containing the parameters to be updated;
        '''

        # update learning rate
        if self.decay != 0:
            self.calcCurrentLearningRate()
        
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weightCache'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasMomentums = np.zeros_like(layer.biases)
            layer.biasCache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weightMomentums = self.beta_1*layer.weightMomentums + (1-self.beta_1)*layer.dweights
        layer.biasMomentums = self.beta_1*layer.biasMomentums + (1-self.beta_1)*layer.dbiases
        # Get corrected momentum
        weightMomentumsCorrected = layer.weightMomentums/(1-self.beta_1**(self.iterations+1))
        biasMomentumsCorrected = layer.biasMomentums/(1-self.beta_1**(self.iterations+1))

        # Update cache with squared current gradients
        layer.weightCache = self.beta_2*layer.weightCache + (1-self.beta_2)*layer.dweights**2
        layer.biasCache = self.beta_2*layer.biasCache + (1-self.beta_2)*layer.dbiases**2
        # Get corrected cache
        weightCacheCorrected = layer.weightCache/(1-self.beta_2**(self.iterations + 1))
        biasCacheCorrected = layer.biasCache/(1-self.beta_2**(self.iterations + 1))

        # Update the parameters
        layer.weights -= self.currentLearningRate*weightMomentumsCorrected/(np.sqrt(weightCacheCorrected)+self.epsilon)
        layer.biases -= self.currentLearningRate*biasMomentumsCorrected/(np.sqrt(biasCacheCorrected)+self.epsilon)

        # Update the iterations' counter
        self.iterations += 1