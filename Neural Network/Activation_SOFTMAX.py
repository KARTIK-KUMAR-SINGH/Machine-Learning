import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# x = [[1 , 2 , 3 , 3.5] , 
#      [2 , 5 , -1 , 2] , 
#      [-1.5 , 2.7 , 3.3 , -0.8]]

nnfs.init()
class Layer_dense :
    def __init__(self , n_inputs , n_neurons) :
        self.weights = 0.10 * np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self , inputs) :
        self.output = np.dot(inputs , self.weights) + self.biases

class Activation_RELU :
    def forward(self , inputs) :
        self.output = np.maximum(0 , inputs)

class Activation_SOFTMAX :
    def forward(self , inputs) :
        exp_values = np.exp(inputs - np.max(inputs , axis = 1 , keepdims = True))
        norm_values = exp_values / np.sum(exp_values , axis = 1 , keepdims = True)
        self.output = norm_values

X , y = spiral_data(samples = 100 , classes = 3)

dense1 = Layer_dense(2 , 3)
activation1 = Activation_RELU()

dense2 = Layer_dense(3 , 3)
activation2 = Activation_SOFTMAX()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("RELU OUTPUT :-")
print(activation1.output[:5])

print("SOFTMAX OUTPUT :-")
print(activation2.output[:5])