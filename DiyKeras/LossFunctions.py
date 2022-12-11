import numpy as np

class Loss:

    # Calculates the empirical loss
    # given model output and ground truth values
    def calculate(self, yTrue, yPred):
        '''
        Calculates the empirical loss on a batch of data.
        Parameters:
        @yTrue: the true values;
        @yPred: the confidence scores;
        Returns:
        @empirical_loss: float, the empirical loss on the batch.
        '''
        # Calculate sample losses
        sample_losses = self.forwardPass(yTrue, yPred)
        # Calculate mean loss
        empirical_loss = np.mean(sample_losses)
        # Return loss
        return empirical_loss

    # Regularization loss calculation
    def regularizationLoss(self, layer):
        '''
        Calculates the forward pass of the regularization loss for a given layer.
        Parameters:
        @layer: layer on which evaluate the regularization loss;
        Returns:
        @regLoss: regularization loss of the given layer (both L1 and L2 depending on layer's settings);
        '''

        regLoss = 0

        # L1 regularization - weights
        if layer.lambda_1w > 0:
            regLoss += layer.lambda_1w*np.sum(np.abs(layer.weights))
        
        # L2 regularization - weights
        if layer.lambda_2w > 0:
            regLoss += layer.lambda_2w*np.sum(layer.weights*layer.weights)

        # L1 regularization - biases
        if layer.lambda_1b> 0:
            regLoss += layer.lambda_1b*np.sum(np.abs(layer.biases))

        # L2 regularization - biases
        if layer.lambda_2b > 0:
            regLoss += layer.lambda_2b*np.sum(layer.biases*layer.biases)

        return regLoss

class CategoricalCrossEntropy(Loss):
    
    def forwardPass(self, yTrue, yPred):
        '''
        Calculates the Categorical Cross Entropy loss function on a batch of predictions.
        Parameters:
        @yTrue: belonging category of the points, can be categorical labels or one-hot vector representation.
        @yPred: batch of confidence scores;
        Returns:
        @negative_log_likelihood: numpy array of shape (nBatch,), loss for each prediction in the batch;
        '''

        # Number of samples in a batch
        samples = len(yPred) if isinstance(yPred,list) else yPred.shape[0]

        # Clip left side from being zero to prevent ln(0)
        # Clip right side from being 1+1e-7 to prevent ln(1+1e-7)<0
        y_pred_clipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # if categorical labels
        if len(yTrue.shape) == 1:
            # select the column with y_true, because in categorical labels it corresponds to the 
            # index of the correct label; row are all selected, hence the use of range(samples)
            correct_confidences = y_pred_clipped[range(samples),yTrue]
        # if Mask values - only for one-hot encoded labels
        elif len(yTrue.shape) == 2:
            # if mask values, categorical cross entropy can be obtained by sum
            # along columns since only the right class will contribute 
            # (all of the other rows have been set to zero by multiplying by y_pred)
            correct_confidences = np.sum(y_pred_clipped*yTrue,axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backwardPass(self, yTrue, yPred):
        '''
        Evaluates the backward pass wrt yPred. Result is stored in public member self.dinputs.
        Parameters:
        @yTrue: array of shape (nBatch,) if belonging category is represented with its index or (nBatch,nCategories) if one-hot, ground truth labels;
        @yPred: array of shape (nBatch, nCategories), confidence scores (SoftMax's output).
        Modifies:
        @self.dinputs: array of shape (nBatch, nCategories), gradient of the loss function wrt yPred;
        '''
        # Number of samples
        nSamples = len(yPred) if isinstance(yPred,list) else yPred.shape[0]

        # Number of labels in every sample (output dimensionality)
        labels = len(yPred[0]) if isinstance(yPred[0],list) else yPred[0].shape[0]

        # If labels are sparse, turn them into one-hot vector
        if len(yTrue.shape) == 1:
            yTrue = np.eye(labels)[yTrue]

        # Calculate gradient
        # 
        self.dinputs = -yTrue / yPred
        # Normalize gradient
        self.dinputs = self.dinputs / nSamples

class Accuracy(Loss):

    def calculate(self,yTrue,yPred):
        '''
        Calculates the Accuracy on a batch of predictions.
        Parameters:
        @yTrue: belonging category of the points, can be categorical labels or one-hot vector representation.
        @yPred: batch of confidence scores;
        Returns:
        @accuracy: float, the percentage of correct predictions;
        '''

        # Predicted class is the index of the maximum of the confidence scores vector
        predictions = np.argmax(yPred, axis=1)
        # If targets are one-hot encoded - convert them
        # Similarly to predicted class,
        # the belonging category is the index of the maximum of the one-hot vector
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1)
        # True evaluates to 1; False to 0
        accuracy = np.mean(predictions == yTrue)
        return accuracy

# Binary cross-entropy loss
class BinaryCrossentropy(Loss):

    def forwardPass(self, yTrue, yPred):
        '''
        Calculates the Binary Cross Entropy loss function on a batch of predictions.
        Parameters:
        @yTrue: belonging category of the points;
        @yPred: ones or zeros, predicted belonging category of the points;
        Returns:
        @sampleLosses: numpy array of shape (nBatch,), loss for each prediction in the batch;
        '''
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sampleLosses = -(yTrue * np.log(yPredClipped) + (1 - yTrue) * np.log(1 - yPredClipped))
        sampleLosses = np.mean(sampleLosses, axis=-1)
        # Return losses
        return sampleLosses
    
    def backwardPass(self, yTrue, yPred):
        '''
        Evaluates the backward pass wrt yPred. Result is stored in public member self.dinputs.
        Parameters:
        @yTrue: array of shape (nBatch,), belonging category of the points;
        @yPred: array of shape (nBatch,), predicted belonging category of the points.
        Modifies:
        @self.dinputs: array of shape (nBatch, nCategories), gradient of the loss function wrt yPred;
        '''
        # Number of samples
        samples = len(yPred) if isinstance(yPred,list) else yPred.shape[0]
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(yPred[0]) if isinstance(yPred[0],list) else yPred[0].shape[0]
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clippedDvalues = np.clip(yPred, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(yTrue / clippedDvalues - (1 - yTrue) / (1 - clippedDvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error loss
class MSE(Loss): # L2 loss

    def forwardPass(self, yTrue, yPred):
        '''
        Calculates the MSE loss function on a batch of predictions.
        Parameters:
        @yTrue: belonging category of the points;
        @yPred: predictions on the points;
        Returns:
        @sampleLosses: numpy array of shape (nBatch,), loss for each prediction in the batch;
        '''
        # Calculate loss
        sampleLosses = np.mean((yTrue - yPred)**2, axis=-1)
        # Return losses
        return sampleLosses

    def backwardPass(self, yTrue, yPred):
        '''
        Evaluates the backward pass wrt yPred. Result is stored in public member self.dinputs.
        Parameters:
        @yTrue: array of shape (nBatch,), belonging category of the points;
        @yPred: array of shape (nBatch,), predicted belonging category of the points.
        Modifies:
        @self.dinputs: array of shape (nBatch, nCategories), gradient of the loss function wrt yPred;
        '''
        # Number of samples
        samples = len(yPred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(yPred[0])
        # Gradient on values
        self.dinputs = -2 * (yTrue - yPred) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss

    def forward(self, yTrue, yPred):
        '''
        Calculates the MAE loss function on a batch of predictions.
        Parameters:
        @yTrue: belonging category of the points;
        @yPred: predictions on the points;
        Returns:
        @sampleLosses: numpy array of shape (nBatch,), loss for each prediction in the batch;
        '''
        # Calculate loss
        sampleLosses = np.mean(np.abs(yTrue - yPred), axis=-1)
        # Return losses
        return sampleLosses

    def backward(self, yTrue, yPred):
        '''
        Evaluates the backward pass wrt yPred. Result is stored in public member self.dinputs.
        Parameters:
        @yTrue: array of shape (nBatch,), belonging category of the points;
        @yPred: array of shape (nBatch,), predicted belonging category of the points.
        Modifies:
        @self.dinputs: array of shape (nBatch, nCategories), gradient of the loss function wrt yPred;
        '''
        # Number of samples
        samples = len(yPred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(yPred[0])
        # Calculate gradient
        self.dinputs = np.sign(yTrue - yPred) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples