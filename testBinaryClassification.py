import numpy as np
from DiyKeras.Layers import DenseLayer
from DiyKeras.ActivationFunctions import ReLU,Sigmoid
from DiyKeras.LossFunctions import BinaryCrossentropy
from DiyKeras.Optimizers import MomentumGradientDescent,AdaGrad,RMSProp,Adam,DecayGradientDescent
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt

if __name__=="__main__":
    # Create dataset
    X, y = spiral_data(samples=100, classes=2)
    # Reshape labels to be a list of lists
    # Inner list contains one output (either 0 or 1)
    # per each output neuron, 1 in this case
    y = y.reshape(-1, 1)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = DenseLayer(2, 64, lambda_2w=5e-4, lambda_2b=5e-4)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 1 output value
    dense2 = DenseLayer(64, 1)
    # Create Sigmoid activation:
    activation2 = Sigmoid()
    # Create loss function
    loss_function = BinaryCrossentropy()
    # Create optimizer
    optimizer = Adam(decay=5e-7)
    losses = []
    # Train in loop
    for epoch in range(10001):
        # Perform a forward pass of our training data through this layer
        dense1.forwardPass(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forwardPass(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function
        # of first layer as inputs
        dense2.forwardPass(activation1.output)
        # Perform a forward pass through activation function
        # takes the output of second dense layer here
        activation2.forwardPass(dense2.output)
        # Calculate the data loss
        data_loss = loss_function.calculate(y,activation2.output)
        losses.append(data_loss)
        # Calculate regularization penalty
        regularization_loss = loss_function.regularizationLoss(dense1) + loss_function.regularizationLoss(dense2)
        # Calculate overall loss
        loss = data_loss + regularization_loss
        # Calculate accuracy from output of activation2 and targets
        # Part in the brackets returns a binary mask - array consisting
        # of True/False values, multiplying it by 1 changes it into array
        # of 1s and 0s
        predictions = (activation2.output > 0.5) * 1
        accuracy = np.mean(predictions == y)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, '+
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularization_loss:.3f}), ' +
            f'lr: {optimizer.currentLearningRate}')
        # Backward pass
        loss_function.backwardPass(y,activation2.output)
        activation2.backwardPass(loss_function.dinputs)
        dense2.backwardPass(activation2.dinputs)
        activation1.backwardPass(dense2.dinputs)
        dense1.backwardPass(activation1.dinputs)
        # Update weights and biases
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)