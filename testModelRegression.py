from nnfs.datasets import sine_data

from DiyKeras.Model import Model
from DiyKeras.Layers import DenseLayer
from DiyKeras.ActivationFunctions4Model import ReLU, Linear
from DiyKeras.LossFunctions4Model import MSE
from DiyKeras.Optimizers import Adam
from DiyKeras.Accuracy4Model import AccuracyRegression

# Create dataset
X, y = sine_data()
# Instantiate the model
model = Model()
# Add layers
model.add(DenseLayer(1, 64))
model.add(ReLU())
model.add(DenseLayer(64, 64))
model.add(ReLU())
model.add(DenseLayer(64, 1))
model.add(Linear())
# Set loss, optimizer and accuracy objects
model.set(
    loss=MSE(),
    optimizer=Adam(learningRate=0.005, decay=1e-3),
    accuracy=AccuracyRegression()
    )
# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, printEvery=100)