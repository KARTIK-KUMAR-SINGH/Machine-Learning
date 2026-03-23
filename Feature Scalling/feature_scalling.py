# ================================
# IMPORTS
# ================================
import numpy as np
import matplotlib.pyplot as plt
import copy


# ================================
# DATASET
# ================================
# Features: [size, bedrooms, floors, age]
X_train = np.array([
[2104,5,1,45],[1416,3,2,40],[852,2,1,35],[1200,3,1,20],[1800,4,2,15],
[1600,3,2,30],[2400,5,2,10],[1000,2,1,50],[2000,4,2,25],[900,2,1,40]
])

# Target values (house prices)
y_train = np.array([460,232,178,250,380,300,500,150,420,200])


# ================================
# FEATURE SCALING FUNCTION
# ================================
def feature_scaling(X):
    """
    Standardize features using mean and standard deviation

    Why?
    -----
    Different features have different ranges (e.g., size=2000, bedrooms=3)
    Scaling makes gradient descent faster and stable.

    Returns:
        X_scaled : scaled dataset
        mu       : mean of each feature
        sigma    : std deviation of each feature
    """
    mu = np.mean(X, axis=0)       # mean of each column
    sigma = np.std(X, axis=0)     # std deviation of each column

    # Avoid division by zero
    sigma[sigma == 0] = 1

    X_scaled = (X - mu) / sigma
    return X_scaled, mu, sigma


# ================================
# PREDICTION FUNCTION
# ================================
def predict(X, w, b):
    """
    Predict output for multiple examples

    X : (m,n) feature matrix
    w : (n,) weights
    b : scalar bias

    Returns:
        predictions : (m,)
    """
    return np.dot(X, w) + b


# ================================
# COST FUNCTION (MSE)
# ================================
def compute_cost(X, y, w, b):
    """
    Compute Mean Squared Error cost

    Formula:
        J = (1/2m) * sum((f(x) - y)^2)

    Purpose:
        Measures how far predictions are from actual values
    """
    m = X.shape[0]

    predictions = predict(X, w, b)
    error = predictions - y

    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost


# ================================
# GRADIENT FUNCTION (VECTORIZED)
# ================================
def compute_gradient(X, y, w, b):
    """
    Compute gradients of cost w.r.t w and b

    Why?
    -----
    Gradients tell us how to update parameters to reduce error

    Returns:
        dj_dw : gradient for weights
        dj_db : gradient for bias
    """
    m = X.shape[0]

    predictions = predict(X, w, b)
    error = predictions - y

    # Vectorized gradient calculation (FAST)
    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)

    return dj_db, dj_dw


# ================================
# GRADIENT DESCENT
# ================================
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Perform gradient descent to learn w and b

    alpha      : learning rate
    num_iters  : number of iterations

    Returns:
        w, b     : optimized parameters
        J_history: cost over iterations
    """
    w = copy.deepcopy(w_in)
    b = b_in

    J_history = []

    for i in range(num_iters):

        # Compute gradients
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Compute cost for monitoring
        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        # Print progress every 10%
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, J_history


# ================================
# SCALE FEATURES BEFORE TRAINING
# ================================
X_train, mu, sigma = feature_scaling(X_train)


# ================================
# INITIALIZE PARAMETERS
# ================================
initial_w = np.zeros(X_train.shape[1])
initial_b = 0.0


# ================================
# TRAIN MODEL
# ================================
alpha = 0.01       # learning rate (works well after scaling)
iterations = 500000  # fewer iterations needed

w_final, b_final, J_hist = gradient_descent(
    X_train, y_train, initial_w, initial_b, alpha, iterations
)


# ================================
# RESULTS
# ================================
print("\nFinal Parameters:")
print("w =", w_final)
print("b =", b_final)


# ================================
# PREDICTION ON TRAINING DATA
# ================================
print("\nPredictions on training data:")

preds = predict(X_train, w_final, b_final)

for i in range(len(X_train)):
    print(f"Predicted: {preds[i]:.2f}, Actual: {y_train[i]}")


# ================================
# PREDICT NEW DATA (IMPORTANT)
# ================================
def predict_new(x, w, b, mu, sigma):
    """
    Predict for a new example

    IMPORTANT:
    Use SAME mu and sigma from training data
    """
    x = (x - mu) / sigma
    return np.dot(x, w) + b


# Example prediction
x_new = np.array([2000, 3, 2, 20])
price = predict_new(x_new, w_final, b_final, mu, sigma)

print("\nPrediction for new house:", price)


# ================================
# PLOT COST FUNCTION
# ================================
plt.plot(J_hist)
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()