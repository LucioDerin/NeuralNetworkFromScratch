# Forward pass of a batch of @nBatch input vectors of dimension @nInput through two layers
# of @layerWidth1 and @layerWidth2 neurons
# Weights, biases and inputs are randomly generated

# Input dimension: (nBatch,nInput)
# Weights1 dimension: (layerWidth1,nInput)
# Weights2 dimension: (layerWidth2,layerWidth1)
# Bias dimension: (layerWidth_i,)
# Output dimension: (nBatch,layerWidth2)

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

#***** FIRST LAYER *****# 
print("\n**First layer**")

# number of neurons in the current layer
layerWidth1 = 3
# weights of the neurons of this layer
weights1 = np.random.normal(0,1,size=(layerWidth1,nInput))
print("--------")
print("Weights1:")
print(weights1)
print("Shape:",weights1.shape)

# biases of the neurons of this layer
biases1 = np.random.random(layerWidth1)
print("--------")
print("Biases1:")
print(biases1)
print("Shape:", biases1.shape)
# Output: X * W^T + b
output1 = inputs@weights1.T + biases1

print("--------")
print("Output from first layer:")
print(output1)
print("Shape:", output1.shape)

#***** SECOND LAYER *****# 
print("\n**Second layer**")
# number of neurons in the current layer
layerWidth2 = 2
# weights of the neurons of this layer
weights2 = np.random.normal(0,1,size=(layerWidth2,layerWidth1))
print("--------")
print("Weights2:")
print(weights2)
print("Shape:",weights2.shape)

# biases of the neurons of this layer
biases2 = np.random.random(layerWidth2)
print("--------")
print("Biases2:")
print(biases2)
print("Shape:", biases2.shape)
# Output: X * W^T + b
output2 = output1@weights2.T + biases2

print("--------")
print("Output from second layer:")
print(output2)
print("Shape:", output2.shape)