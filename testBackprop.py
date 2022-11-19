from timeit import timeit
import numpy as np
from DiyKeras.ActivationFunctions import ActivationSoftmaxCategoricalCrossEntropy,ActivationSoftmax
from DiyKeras.LossFunctions import CategoricalCrossEntropy

def f1():
    softmax_loss = ActivationSoftmaxCategoricalCrossEntropy()
    softmax_loss.backwardPass(class_targets, softmax_outputs)
    dvalues1 = softmax_loss.dinputs

def f2():
    activation = ActivationSoftmax()
    activation.output = softmax_outputs
    loss = CategoricalCrossEntropy()
    loss.backwardPass(class_targets, softmax_outputs)
    activation.backwardPass(loss.dinputs)
    dvalues2 = activation.dinputs

def f1Return():
    softmax_loss = ActivationSoftmaxCategoricalCrossEntropy()
    softmax_loss.backwardPass(class_targets, softmax_outputs)
    dvalues1 = softmax_loss.dinputs
    return dvalues1

def f2Return():
    activation = ActivationSoftmax()
    activation.output = softmax_outputs
    loss = CategoricalCrossEntropy()
    loss.backwardPass(class_targets, softmax_outputs)
    activation.backwardPass(loss.dinputs)
    dvalues2 = activation.dinputs
    return dvalues2


softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])

print("Execution time ratio between slow and fast implementation of softmax+CE gradient:")
t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2/t1)

print("Results (should be equal):")
print(f1Return())
print(f2Return())