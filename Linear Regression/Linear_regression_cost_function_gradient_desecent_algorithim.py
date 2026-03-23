import numpy as np 
#import matplotlib.pyplot as plt 



def gradient_descent(x , y , w , b , alpha , iterations) :
    cost_history = []
    for i in range (iterations) :
        dj_dw , dj_db = compute_gradient(x , y , w , b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x , y , w , b)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"iterations {i} Cost: {cost}")
    return w , b , cost_history

def compute_gradient(x , y , w , b) :
    m = len(x)

    dj_dw = 0
    dj_db = 0 

    for i in range(m) :
        f = w*x[i] + b

        dj_dw += (f - y[i]) * x[i]
        dj_db += (f - y[i])

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw  , dj_db

def compute_cost(x , y , w , b) :
    m = len(x)
    cost = 0

    for i in range(m):
        f = w * x[i] + b
        cost = cost + (f - y[i])** 2
    total_cost = cost / (2*m)
    return total_cost

x_train = np.array([1, 2, 3, 4])
y_train = np.array([300, 500, 700, 900])

w_init = 0
b_init = 0

alpha = 0.01
iterations = 1000

w_final, b_final, cost_hisotry = gradient_descent(x_train , y_train , w_init , b_init , alpha , iterations)

print("\nFinal Parametres")
print("w = " , w_final)
print("b = " , b_final)

size = 2.5
predicted_price = w_final * size + b_final
print("\nPredicted price for 2500 sqft house: " , predicted_price , "thousand $")