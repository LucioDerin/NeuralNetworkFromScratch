import numpy as np

class Loss:

    # Calculates the empirical loss
    # given model output and ground truth values
    def calculate(self, yTrues, yPreds):
        '''
        Calculates the empirical loss on a batch of data.
        Parameters:
        @yTrues: the true values;
        @yPreds: the confidence scores;
        Returns:
        @empirical_loss: float, the empirical loss on the batch.
        '''
        # Calculate sample losses
        sample_losses = self.forwardPass(yTrues, yPreds)
        # Calculate mean loss
        empirical_loss = np.mean(sample_losses)
        # Return loss
        return empirical_loss

class CategoricalCrossEntropy(Loss):
    
    def forwardPass(self, yTrues, yPreds):
        '''
        Calculates the Categorical Cross Entropy loss function on a batch of predictions.
        Parameters:
        @yTrues: belonging category of the points, can be categorical labels or one-hot vector representation.
        @yPreds: batch of confidence scores;
        Returns:
        @negative_log_likelihood: numpy array of shape (nBatch,), loss for each prediction in the batch;
        '''

        # Number of samples in a batch
        samples = len(yPreds) if isinstance(yPreds,list) else yPreds.shape[0]

        # Clip left side from being zero to prevent ln(0)
        # Clip right side from being 1+1e-7 to prevent ln(1+1e-7)<0
        y_pred_clipped = np.clip(yPreds, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # if categorical labels
        if len(yTrues.shape) == 1:
            # select the column with y_true, because in categorical labels it corresponds to the 
            # index of the correct label; row are all selected, hence the use of range(samples)
            correct_confidences = y_pred_clipped[range(samples),yTrues]
        # if Mask values - only for one-hot encoded labels
        elif len(yTrues.shape) == 2:
            # if mask values, categorical cross entropy can be obtained by sum
            # along columns since only the right class will contribute 
            # (all of the other rows have been set to zero by multiplying by y_pred)
            correct_confidences = np.sum(y_pred_clipped*yTrues,axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Accuracy(Loss):

    def calculate(self,yTrues,yPreds):
        '''
        Calculates the Accuracy on a batch of predictions.
        Parameters:
        @yTrues: belonging category of the points, can be categorical labels or one-hot vector representation.
        @yPreds: batch of confidence scores;
        Returns:
        @accuracy: float, the percentage of correct predictions;
        '''

        # Predicted class is the index of the maximum of the confidence scores vector
        predictions = np.argmax(yPreds, axis=1)
        # If targets are one-hot encoded - convert them
        # Similarly to predicted class,
        # the belonging category is the index of the maximum of the one-hot vector
        if len(yTrues.shape) == 2:
            yTrues = np.argmax(yTrues, axis=1)
        # True evaluates to 1; False to 0
        accuracy = np.mean(predictions == yTrues)
        return accuracy