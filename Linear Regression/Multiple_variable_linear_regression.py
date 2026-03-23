# ================================
# IMPORTS
# ================================
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

# ================================
# DATA
# ================================
# Features: [size, bedrooms, floors, age]
X_train = np.array([
[2104,5,1,45],[1416,3,2,40],[852,2,1,35],[1200,3,1,20],[1800,4,2,15],
[1600,3,2,30],[2400,5,2,10],[1000,2,1,50],[2000,4,2,25],[900,2,1,40],
[2200,4,2,20],[1300,3,1,30],[1700,4,2,18],[1100,2,1,45],[1500,3,2,35],
[2600,5,2,12],[800,1,1,55],[1900,4,2,28],[1400,3,1,33],[2300,5,2,14],
[1250,3,1,22],[1750,4,2,17],[950,2,1,48],[2050,4,2,24],[1350,3,1,29],
[1650,4,2,26],[850,2,1,52],[1950,4,2,23],[1450,3,1,31],[2150,5,2,19],
[1550,3,2,27],[1050,2,1,44],[1850,4,2,21],[1150,2,1,46],[2250,5,2,16],
[1000,2,1,38],[2100,4,2,20],[1200,3,1,28],[1800,4,2,15],[1400,3,1,32],
[1600,3,2,30],[2400,5,2,10],[900,2,1,40],[2000,4,2,25],[1300,3,1,35],
[1700,4,2,18],[1100,2,1,45],[1500,3,2,35],[2600,5,2,12],[800,1,1,55],
[1900,4,2,28],[1400,3,1,33],[2300,5,2,14],[1250,3,1,22],[1750,4,2,17],
[950,2,1,48],[2050,4,2,24],[1350,3,1,29],[1650,4,2,26],[850,2,1,52],
[1950,4,2,23],[1450,3,1,31],[2150,5,2,19],[1550,3,2,27],[1050,2,1,44],
[1850,4,2,21],[1150,2,1,46],[2250,5,2,16],[1000,2,1,38],[2100,4,2,20],
[1200,3,1,28],[1800,4,2,15],[1400,3,1,32],[1600,3,2,30],[2400,5,2,10],
[900,2,1,40],[2000,4,2,25],[1300,3,1,35],[1700,4,2,18],[1100,2,1,45],
[1500,3,2,35],[2600,5,2,12],[800,1,1,55],[1900,4,2,28],[1400,3,1,33],
[2300,5,2,14],[1250,3,1,22],[1750,4,2,17],[950,2,1,48],[2050,4,2,24]
])

y_train = np.array([
460,232,178,250,380,
300,500,150,420,200,
450,260,390,180,310,
520,120,410,270,490,
255,395,170,430,275,
305,160,415,285,460,
320,190,405,200,480,
210,440,260,380,270,
300,500,200,420,280,
390,180,310,520,120,
410,270,490,255,395,
170,430,275,305,160,
415,285,460,320,190,
405,200,480,210,440,
260,380,270,300,500,
200,420,280,390,180,
310,520,120,410,270,
490,255,395,170,430
])

# ================================
# PREDICTION FUNCTION
# ================================
def predict(x, w, b):
    """
    Predict output for a single example
    x: (n,)
    w: (n,)
    b: scalar
    """
    return np.dot(x, w) + b


# ================================
# COST FUNCTION
# ================================
def compute_cost(X, y, w, b):
    """
    Compute Mean Squared Error cost
    X: (m,n)
    y: (m,)
    w: (n,)
    b: scalar
    """
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i]) ** 2

    cost = cost / (2 * m)
    return cost


# ================================
# GRADIENT FUNCTION
# ================================
def compute_gradient(X, y, w, b):
    """
    Compute gradients for w and b
    """
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] += error * X[i, j]

        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


# ================================
# GRADIENT DESCENT
# ================================
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):

    w = copy.deepcopy(w_in)
    b = b_in

    J_history = []

    for i in range(num_iters):

        # Compute gradients
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost
        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        # Print progress
        if i % max(1, num_iters // 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost:.4f}")

    return w, b, J_history


# ================================
# TRAIN MODEL
# ================================
# Initialize parameters
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.0

# Hyperparameters
iterations = 100000
alpha = 5e-7

# Run gradient descent
w_final, b_final, J_hist = gradient_descent(
    X_train, y_train, initial_w, initial_b, alpha, iterations
)

# ================================
# RESULTS
# ================================
print("\nFinal Parameters:")
print("w =", w_final)
print("b =", b_final)

print("\nPredictions:")
for i in range(len(X_train)):
    prediction = np.dot(X_train[i], w_final) + b_final
    print(f"Predicted: {prediction:.2f}, Actual: {y_train[i]}")


# ================================
# PLOT COST
# ================================
plt.plot(J_hist)
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()