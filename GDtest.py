from timeit import timeit
import numpy as np
from DiyKeras.DenseLayer import DenseLayer
from DiyKeras.ActivationFunctions import ActivationSoftmaxCategoricalCrossEntropy,ActivationSoftmax,ActivationReLU
from DiyKeras.LossFunctions import CategoricalCrossEntropy
from DiyKeras.Optimizers import NesterovGradientdescent
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    Y = np.copy(y)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = DenseLayer(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = ActivationReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = DenseLayer(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = ActivationSoftmaxCategoricalCrossEntropy()
    # Create optimizer
    optimizer = NesterovGradientdescent(momentum=0.9,decay=1e-3)

    iter = []
    losses = []
    accs = []
    lr = []
    # Train in loop
    for epoch in range(10000):
        iter.append(epoch)
        # Perform a forward pass of our training data through this layer
        dense1.forwardPass(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forwardPass(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forwardPass(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forwardPass(y, dense2.output)
        losses.append(loss)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        accs.append(accuracy)
        lr.append(optimizer.currentLearningRate)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}',end="\r")
        # Backward pass
        loss_activation.backwardPass(y, loss_activation.output)
        dense2.backwardPass(loss_activation.dinputs)
        activation1.backwardPass(dense2.dinputs)
        dense1.backwardPass(activation1.dinputs)
        # Update weights and biases
        optimizer.updateParams(dense1)
        optimizer.updateParams(dense2)
    plt.subplot(121)
    plt.plot(iter,losses,label='Training Loss')
    plt.plot(iter,accs,label='Training Accuracy')
    plt.plot(iter,lr,label='Learning Rate')
    plt.xlabel("#Iterations")
    plt.title("Training")
    plt.legend(loc='best')
    #plt.semilogy()

    Ngrid = 100
    x = np.linspace(X[:,0].min(),X[:,0].max(),Ngrid)
    y = np.linspace(X[:,1].min(),X[:,1].max(),Ngrid)
    
    xx,yy = np.meshgrid(x,y)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    # Perform a forward pass of our training data through this layer
    dense1.forwardPass(grid)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forwardPass(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forwardPass(activation1.output)
    predGrid = np.argmax(dense2.output,axis=1)

    z = predGrid.reshape(Ngrid,Ngrid)
    plt.subplot(122)
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='brg')
    plt.pcolormesh(x,y,z,shading='auto',cmap = 'brg',alpha=0.2)
    plt.title("Separation")

    plt.show()