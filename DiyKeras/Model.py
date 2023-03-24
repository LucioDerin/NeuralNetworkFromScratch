from DiyKeras.Layers import InputLayer

# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.inputLayer = InputLayer()
        # Count all the objects
        nLayers = len(self.layers)
        # Initialize a list containing trainable layers:
        self.trainableLayers = []

        # Iterate the objects
        for i in range(nLayers):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.inputLayer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and the last
            elif i < nLayers - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.outputLayerActivation = self.layers[i]
            
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainableLayers.append(self.layers[i])
        # Update loss object with trainable layers
        self.loss.rememberTrainableLayers(self.trainableLayers)

    # Performs forward pass
    def forwardPass(self, X):
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.inputLayer.forwardPass(X)
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forwardPass(layer.prev.output)

        # return last layer's output
        return self.layers[-1].output

    # Performs backward pass
    def backwardPass(self, y, output):
        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backwardPass(y, output)
        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backwardPass(layer.next.dinputs)

    # Train the model
    def train(self, X, y, *, epochs=1, printEvery=1, validationData=None):
        # Initialize accuracy object
        self.accuracy.init(y)
        # Main training loop
        for epoch in range(1, epochs+1):
            # Perform the forward pass
            output = self.forwardPass(X)
            # Calculate loss
            dataLoss, regularizationLoss = self.loss.calculate(y,output,includeRegularization=True)
            loss = dataLoss + regularizationLoss
            # Get predictions and calculate an accuracy
            predictions = self.outputLayerActivation.predictions(output)
            accuracy = self.accuracy.calculate(y, predictions)

            # Perform backward pass
            self.backwardPass(y, output)
            # Optimize (update parameters)
            for layer in self.trainableLayers:
                self.optimizer.updateParams(layer)
            
            # Print a summary
            if not epoch % printEvery:
                print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + 
                f'loss: {loss:.3f} (' + 
                f'data_loss: {dataLoss:.3f}, ' +
                f'reg_loss: {regularizationLoss:.3f}), ' +
                f'lr: {self.optimizer.currentLearningRate}')

            if validationData is not None:
                # For better readability
                Xval, yVal = validationData
                # Perform the forward pass
                output = self.forwardPass(Xval)
                # Calculate the loss
                loss = self.loss.calculate(yVal, output)
                # Get predictions and calculate an accuracy
                predictions = self.outputLayerActivation.predictions(output)
                accuracy = self.accuracy.calculate(yVal, predictions)
                # Print a summary
                print(f'validation, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')