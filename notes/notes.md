## Notes:

### Files description:
- `singleLayerForwardPass.py`: forward pass of a single input vector;
- `singleLayerForwardPassBatchOfInputs.py`: forward pass of a batch of input data points;
- `twoLayersForwardPassBatchOfInputs.py`: forward pass of a batch of data through two layers;

### General Notes:

#### Batch of data
A batch of data from the feature space is contained in the matrix $\hat{X}$. Even though each sample in the batch is a vector in the feature space, it is common practice to treat them as row vectors inside the matrix, so that $\hat{X}$ will be an $n\times d$ matrix (with $n$ number of samples and $d$ dimensionality of the feature space). Along with this choice, for any given matrix, it is common practice to let the first index run through the cardinality of the sample, and treat vectors as row vectors. For instance, the matrix of neuron's weights will have the first index selecting the i-th neuron and the second index selecting the j-th weight of the i-th neuron. The matrix $W$ will be $w \times d$ with $w$ width of the layer (number of neurons) and $d$ dimensionality of the input.

With this choices, the action of a layer on the input batch is:
$$\hat{Y} = \hat{X}W^{T} + b$$

$\hat{X}W^{T}$ is a matrix product between matrix of shapes $n\times d$ and $d \times w$ resulting in a matrix $n \times w$; the first index of the output matrix points to the output of the i-th data point, and the second index runs through the dimensionality of the output space.

**Note:** the direct sum of $\hat{X}W^{T}$ and $b$ is not mathematically well defined, since b is a row vector of dimension $w$ and $\hat{X}W^{T}$ is a matrix of dimension $n \times w$. Nevertheless, in this case, `numpy` sums b to each row of $\hat{X}W^{T}$ which is exactly how a bias behaves.

### Multiple Layers:
The dimensionality of the weights of the next layer is determined by the previous one, while the number of neurons is arbitrary.
So, let $W_i$ be the weights' matrix of the i-th layer and $w_i$ the width of the i-th layer (e.g. the number of neurons in the i-th layer): then $W$ will have dimension $w_{i-1} \times w_i$,