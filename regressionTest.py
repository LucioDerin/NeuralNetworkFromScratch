import numpy as np
from matplotlib import pyplot as plt
from nnfs.datasets import sine_data
from DiyKeras.Layers import DenseLayer
from DiyKeras.ActivationFunctions import ReLU,Linear
from DiyKeras.LossFunctions import MSE
from DiyKeras.Optimizers import Adam


# Create dataset
X, y = sine_data()
# Create Dense layer with 1 input feature and 64 output values
dense1 = DenseLayer(1, 64)
dense1.scale = 0.1
dense1.epsilon = 1
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 64 output values
dense2 = DenseLayer(64, 64)
dense2.scale = 0.1
dense2.epsilon = 1
# Create ReLU activation (to be used with Dense layer):
activation2 = ReLU()
# Create third Dense layer with 64 input features (as we take output
# of previous layer here) and 1 output value
dense3 = DenseLayer(64, 1)
dense3.scale = 0.1
dense3.epsilon = 1

# Create Linear activation:
activation3 = Linear()
# Create loss function
loss_function = MSE()
# Create optimizer
optimizer = Adam(learningRate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of all the ground truth values
accuracy_precision = np.std(y) / 250
# Train in loop
import matplotlib.pyplot as plt
X_test, y_test = sine_data()
plt.figure()

for epoch in range(1001):
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
    # Perform a forward pass through third Dense layer
    # takes outputs of activation function of second layer as inputs
    dense3.forwardPass(activation2.output)
    # Perform a forward pass through activation function
    # takes the output of third dense layer here
    activation3.forwardPass(dense3.output)
    # Calculate the data loss
    data_loss = loss_function.calculate(activation3.output, y)
    # Calculate regularization penalty
    regularization_loss = \
    loss_function.regularizationLoss(dense1) + \
    loss_function.regularizationLoss(dense2) + \
    loss_function.regularizationLoss(dense3)
    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    # Backward pass
    loss_function.backwardPass(y, activation3.output)
    activation3.backwardPass(loss_function.dinputs)
    dense3.backwardPass(activation3.dinputs)
    activation2.backwardPass(dense3.dinputs)
    dense2.backwardPass(activation2.dinputs)
    activation1.backwardPass(dense2.dinputs)
    dense1.backwardPass(activation1.dinputs)
    # Update weights and biases
    optimizer.updateParams(dense1)
    optimizer.updateParams(dense2)
    optimizer.updateParams(dense3)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f} (' +
        f'data_loss: {data_loss:.3f}, ' +
        f'reg_loss: {regularization_loss:.3f}), ' +
        f'lr: {optimizer.currentLearningRate}')
        dense1.forwardPass(X_test)
        activation1.forwardPass(dense1.output)
        dense2.forwardPass(activation1.output)
        activation2.forwardPass(dense2.output)
        dense3.forwardPass(activation2.output)
        activation3.forwardPass(dense3.output)
        plt.plot(X_test, y_test,label='Real')
        plt.plot(X_test,activation3.output,label='Predicted')
        plt.legend(loc='best')
        plt.pause(0.05)
        plt.clf()

plt.show()