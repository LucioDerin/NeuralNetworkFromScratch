# Notes

### Notation, conventions, and other stuff
- NN = Neural Networks, ML = Machine Learning, SV = Supervised Learning;
- I adopt a notation (taught to me by my uni professor) that I find very useful in ML, which is to denote with a hat all of the quantities that depend on the dataset (empirical quantities), e.g. $\hat{X}$;
- I name every variable or function with lower camel case, every class with upper camel case; file names are in lower camel case except if they contain a class;

### Files description
- `singleLayerForwardPass.py`: forward pass of a single input vector;
- `singleLayerForwardPassBatchOfInputs.py`: forward pass of a batch of input data points;
- `twoLayersForwardPassBatchOfInputs.py`: forward pass of a batch of data through two layers;
- `DenseLayer.py`: contains the class `DenseLayer`;
- `ActivationFunctions.py`: contains the activation functions' classes;
- `LossFunctions.py`: contains the loss functions' classes;
- `test.py`: a small test of all the implemented classes;

## General Notes

### Batch of data
A batch of data from the feature space is contained in the matrix $\hat{X}$. Even though each sample in the batch is a vector in the feature space, it is common practice to treat them as row vectors inside the matrix, so that $\hat{X}$ will be an $n\times d$ matrix (with $n$ number of samples and $d$ dimensionality of the feature space). Along with this choice, for any given matrix, it is common practice to let the first index run through the cardinality of the sample, and treat vectors as row vectors. For instance, the matrix of neuron's weights will have the first index selecting the i-th neuron and the second index selecting the j-th weight of the i-th neuron. The matrix $W$ will be $w \times d$ with $w$ width of the layer (number of neurons) and $d$ dimensionality of the input.

With this choices, the action of a layer on the input batch (neglecting activation functions) is:
$$\hat{Y} = \hat{X}W^{T} + b$$

$\hat{X}W^{T}$ is a matrix product between matrix of shapes $n\times d$ and $d \times w$ resulting in a matrix $n \times w$; the first index of the output matrix points to the output of the i-th data point, and the second index runs through the dimensionality of the output space.
**Note 1:** Later in the book, the authors define the weights' matrix to be of shape $d \times w$ in order to get rid of the transpose operation. This has obvious computational advantages, so I will adopt their choice, making the first index of $W$ run trough the dimensionality of the input space and the second one run through the neurons in the layer. With this choice the forward pass is simply:
$$\hat{Y} = \hat{X}W + b$$
and we avoid a transposition each time.
**Note 2:** the direct sum of $\hat{X}$ and $b$ is not mathematically well defined, since b is a row vector of dimension $w$ and $\hat{X}W$ is a matrix of dimension $n \times w$. Nevertheless, in this case, `numpy` sums b to each row of $\hat{X}W$ which is exactly how a bias behaves.

### Multiple Layers
The dimensionality of the weights of the next layer is determined by the previous one, while the number of neurons is arbitrary.
So, let $W_i$ be the weights' matrix of the i-th layer and $w_i$ the width of the i-th layer (e.g. the number of neurons in the i-th layer): then $W$ will have dimension $w_{i-1} \times w_i$.

### Weights initialization
Usually this process is randomized and weights are initialized with small absolute values, typically sampling a gaussian pdf. Biases are typically initialized to zero. These are not the only or best choices in every scenario, and some considerations can increase the performances of an algorithm:
- Typically optimized weights end up being between $\pm1$; the common choice is to initialize them 
a couple of magnitude order below, because otherwise the training will require a lot of time because the correction at each epoch is small (and the majority of weights will not be near $\pm1$).
- Biases guarantees that a neuron fires even if it's not been activated by the input;
- In cases where a neuron does not fire for any sample in the batch (common case with sparse signals), a non zero bias guarantees that we do not end up in what's called a "dead net", characterized by:
    - One or more neurons in the first layer that are silent for every input;
    - These zero outputs are fed to the next layer, killing other neurons throughout the network;
    - The network becomes untrainable: this is a clear indication that we need to think about different initializations;

### Activation functions
Activation function are deployed on each layer to estimate non linear relations between the input and the output spaces. The forward pass defined as $\hat{X}W$ is linear, so it will only estimate linear target functions. If you add a bias it is non linear, but it's still a very simple model.
The way in which NNs add the non-linearity (as opposed to kernel methods) is by wrapping all the linear model in a non-linearity (the so called activation function):
$$\hat{Y} = \sigma(\hat{X}W + b)$$
This mimics the behavior of a biological neuron.

Typically all hidden layers will have the same activation function (but it's not a must), while the output layer's activation function varies based on the application (regression, binary/multiclass classification).

#### Types of activation functions considered

##### Mostly for hidden layers
- **Heaviside**: easiest activation, the neuron only fires if a certain threshold is reached. It's been deployed historically but nowadays it's dismissed;
- **Linear**: should be called identity since it leaves the forward pass unaltered; it is commonly deployed in the output layer for regression models where one has to estimate a real number potentially unbounded. Nevertheless, if every layer uses the linear activation function, the model will only estimate linear target functions;
- **Sigmoid**: it's meant to be a little more informative than the Heaviside one, by carrying also the information of "how close we were to the activation threshold" (mathematically, it should be invertible). Mathematically, it is the logistic curve defined by:
$$y = \frac{1}{1+e^{-x}}$$
Nowadays it's been replaced by the ReLU (mainly for speed and efficiency), although it's worth to notice that the sigmoid is the natural activation function inside animal's neurons.
- **ReLU**: rectified linear units, it's the most common activation function for hidden layers. Can be defined as:
$$y = \max\{0,x\}$$
By linear combination of ReLUs we are estimating a function with linear interpolation, with a granularity that is proportional to the number of neurons.

##### Mostly for output layer
- **Softmax**: deployed in classification models to get an output which is interpretable as a probability (i.e. normalized) from non normalized inputs. The output of softmax will be a vector of $m$ dimension, with m number of classes in the classification setup. The entries of the vector sum up to one and are the probabilities of the input to belong to the respective class (confidence scores). The analytical formulation is:
$$Y_i = \frac{e^{x_i}}{\sum\limits_{j=1}^{d}e^{x_j}}$$
where $d$ is the dimensionality of the output of the previous layer.
In the implementation, to avoid exploding exponentials, the $x_i$ are firstly shifted back by the largest $x_i$ so that the exponential will have at most 0 as an argument. The normalization guarantees that the final confidence scores are unaltered by this shift.

### Loss Functions

#### Mean Squared Error
Loss function typically deployed in regression (but also suitable for classification).

#### Categorical (Binary) Cross Entropy
Loss function deployed in classification problems, perfect if coupled with softmax in the last layer. It compares the confidence scores predicted by softmax $\hat{y}_i$ with the true labels $y_i$ in the following manner (note: this only refers to one point and the true labels are not $\pm1$ but are converted to vectors with a one in the class in which the point belongs, e.g. with to_categorical):
$$\hat{l}_{CE} = -\sum\limits_{j=1}^{m} y_j\ln(\hat{y}_j) = -\sum\limits_{j|y_j=1}\ln(\hat{y}_j)$$
More precisely, the first sum is called cross-entropy and is a more general functional that compares two vectors containing samples of a distribution. When applied to a vector that indicates in which class the points belong (e.g. $y_i = (1,0,0,0)$), cross-entropy takes the form of the second sum and it's called categorical cross-entropy. If $m=2$ the classes are mutually exclusive and the confidence scores must sum up to one, so categorical cross entropy takes the form:
$$\hat{l}_{CE} = -(y\ln(\hat{y}) + (1-y) \ln(1-\hat{y}))$$
and it is called binary cross entropy.

Categorical (binary) cross-entropy works by pushing the network to maximize the confidence on the true class (it's the only class that enters the sum in the loss, and $\min\{\ln(x)\}|_{x<1}$ is 1). It does not explicitly force the other scores to be zero, but this is achieved tanks to normalization.

**Note:** when using categorical cross-entropy the belonging class of the point must be labelled with 1 and the others to zero, not $\pm1$. This is called **one-hot** encoding.




### Things to try
- Add a little offset to the weights initialization so that no weight is set to zero and check if training convergence changes significantly (biases are set to zero, so in conjunction with a zero weight the neuron won't fire at the beginning of the training); eg instead of doing 
`self.weights = self.scale * np.random.randn(shape=(nInputs,nNeurons))`
do
`self.weights = self.scale * np.random.randn(shape=(nInputs,nNeurons)) + self.epsilon`
- Explicitly force the minimization of the confidence score in the wrong classes in categorical cross-entropy, and see if it speeds up convergence.