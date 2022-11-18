# Forward pass of a batch of @nBatch input vectors of dimension @nInput through a layer of @layerWidth neurons
# Weights, biases and inputs are randomly generated

# Input dimension: (nBatch,nInput)
# Weights dimension: (layerWidth,nInput)
# Bias dimension: (layerWidth,)
# Output dimension: (nBatch,layerWidth)

import numpy as np

# dimensionality of the input
nInput = 4
# number of data points in the batch
nBatch = 3

# inputs of the layer
# could be a vector from the input space if the layer is the input layer
# or a vector of the outputs from the neurons of the previous layer
inputs = np.random.normal(0,2,size=(nBatch,nInput))
print("Inputs:")
print(inputs)
print("Shape:", inputs.shape)

# number of neurons in the current layer
layerWidth = 2
# weights of the neurons of this layer
weights = np.random.normal(0,1,size=(layerWidth,nInput))
print("--------")
print("Weights:")
print(weights)
print("Shape:",weights.shape)

# biases of the neurons of this layer
biases = np.random.random(layerWidth)
print("--------")
print("Biases:")
print(biases)
print("Shape:", biases.shape)
# Output: X * W^T + b
output = inputs@weights.T + biases

print("--------")
print("Output from this layer:")
print(output)
print("Shape:", output.shape)