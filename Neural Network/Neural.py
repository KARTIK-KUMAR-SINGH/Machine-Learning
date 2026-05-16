import numpy as np
inputs = [[1 , 2 , 3 , 4] ,
          [2 , 5 , -6 , -7],
          [9 , 10 , 11 , 12]]
weights = [[0.2 , 0.8 , 0.5 , 0.10] ,
           [0.55 , 0.6 , 0.8 , 0.9] , 
           [-0.26 , -0.27 , 0.17 , 0.87]]
biases = [2 , 3 , 0.5]
output = np.dot(inputs , np.array(weights).T) + biases
print(output)