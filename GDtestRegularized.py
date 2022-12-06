import numpy as np
from DiyKeras.Layers import DenseLayer
from DiyKeras.ActivationFunctions import SoftmaxCategoricalCrossEntropy,Softmax,ReLU
from DiyKeras.LossFunctions import CategoricalCrossEntropy
from DiyKeras.Optimizers import MomentumGradientDescent,AdaGrad,RMSProp,Adam
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    Y = np.copy(y)
    # Create Dense layer with 2 input features and 64 output values
    # and L2 regularization on both weights and biases
    dense1 = DenseLayer(2, 64, lambda_2w=5e-4, lambda_2b=5e-4)

    # Create ReLU activation (to be used with Dense layer):
    activation1 = ReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = DenseLayer(64, 3)
    # Create Softmax classifier's combined loss and activation
    loss_activation = SoftmaxCategoricalCrossEntropy()
    # Create optimizer
    #optimizer = MomentumGradientdescent(momentum=0.9,decay=1e-3)
    #optimizer = AdaGrad(decay=1e-6)
    #optimizer = RMSProp(learningRate=0.02, decay=1e-5,rho=0.999)
    optimizer = Adam(learningRate=0.05, decay=5e-7)

    iter = []
    losses = []
    accs = []
    lr = []

    Ngrid = 100
    xgrid = np.linspace(X[:,0].min(),X[:,0].max(),Ngrid)
    ygrid = np.linspace(X[:,1].min(),X[:,1].max(),Ngrid)
    xx,yy = np.meshgrid(xgrid,ygrid)
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

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
        # regularization terms
        loss += loss_activation.regularizationLoss(dense1)
        loss += loss_activation.regularizationLoss(dense2)

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

        if epoch%100==0 and epoch !=0 and epoch != 10000-1:
            plt.subplot(121)
            plt.plot(iter,losses,label='Training Loss')
            plt.plot(iter,accs,label='Training Accuracy')
            plt.plot(iter,lr,label='Learning Rate')
            plt.xlabel("#Iterations")
            plt.title("Training")
            plt.legend(loc='upper left')

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
            plt.pcolormesh(xgrid,ygrid,z,shading='auto',cmap = 'brg',alpha=0.2)
            plt.title("Separation")
            plt.pause(0.05)
            plt.clf()

    plt.show()