import numpy as np

x = [[1 , 2 , 3 , 3.5] , 
     [2 , 5 , -1 , 2] , 
     [-1.5 , 2.7 , 3.3 , -0.8]]

class Layer_dense :
    def __init__(self , n_inputs , n_neurons) :
        self.weights = 0.10 * np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1 , n_neurons))
    def forward(self , inputs) :
        self.output = np.dot(inputs , self.weights) + self.biases

class Activation_RELU :
    def forward(self , inputs) :
        self.output = np.maximum(0 , inputs)

layer1 = Layer_dense(4 , 5)
activation1 = Activation_RELU()
layer2 = Layer_dense(5 , 2)


layer1.forward(x)
activation1.forward(layer1.output)
layer2.forward(activation1.output)

print(layer2.output)