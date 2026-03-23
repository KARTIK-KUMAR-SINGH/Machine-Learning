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

# Prices
y_train = np.array([460,232,178,250,380,300,500,150,420,200])


# ================================
# STEP 1: POLYNOMIAL FEATURES
# ================================
def polynomial_features(X):
    """
    Convert features into polynomial form (degree = 2)

    Example:
    [size, bedrooms] →
    [size, bedrooms, size², bedrooms²]

    WHY?
    To capture non-linear relationships
    """
    return np.concatenate([X, X**2], axis=1)


# ================================
# STEP 2: FEATURE SCALING
# ================================
def feature_scaling(X):
    """
    Standardization:
    X = (X - mean) / std

    WHY?
    All features become similar scale → faster learning
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    sigma[sigma == 0] = 1

    X_scaled = (X - mu) / sigma
    return X_scaled, mu, sigma


# ================================
# STEP 3: PREDICTION FUNCTION
# ================================
def predict(X, w, b):
    """
    Linear model:
    y = Xw + b
    """
    return np.dot(X, w) + b


# ================================
# STEP 4: COST FUNCTION
# ================================
def compute_cost(X, y, w, b):
    """
    Mean Squared Error:
    J = (1/2m) * sum((pred - actual)^2)

    Measures how wrong the model is
    """
    m = X.shape[0]
    error = predict(X, w, b) - y
    return (1/(2*m)) * np.sum(error**2)


# ================================
# STEP 5: GRADIENT FUNCTION
# ================================
def compute_gradient(X, y, w, b):
    """
    Computes slope of cost function

    WHY?
    To know how to update weights to reduce error
    """
    m = X.shape[0]

    error = predict(X, w, b) - y

    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)

    return dj_db, dj_dw


# ================================
# STEP 6: GRADIENT DESCENT
# ================================
def gradient_descent(X, y, w, b, alpha, iterations):
    """
    Iteratively updates w and b

    alpha → learning rate
    iterations → how many steps
    """
    J_history = []

    for i in range(iterations):

        dj_db, dj_dw = compute_gradient(X, y, w, b)

        # Update rule
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % (iterations // 10) == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, J_history


# ================================
# STEP 7: APPLY TRANSFORMATIONS
# ================================

# Add polynomial features
X_poly = polynomial_features(X_train)

# Scale features
X_poly, mu, sigma = feature_scaling(X_poly)


# ================================
# STEP 8: INITIALIZE PARAMETERS
# ================================
w_init = np.zeros(X_poly.shape[1])
b_init = 0.0


# ================================
# STEP 9: TRAIN MODEL
# ================================
alpha = 0.01
iterations = 50000

w_final, b_final, J_hist = gradient_descent(
    X_poly, y_train, w_init, b_init, alpha, iterations
)


# ================================
# STEP 10: TRAINING PREDICTIONS
# ================================
print("\nPredictions:")
preds = predict(X_poly, w_final, b_final)

for i in range(len(preds)):
    print(f"Predicted: {preds[i]:.2f}, Actual: {y_train[i]}")


# ================================
# STEP 11: PREDICT NEW DATA
# ================================
def predict_new(x, w, b, mu, sigma):
    """
    Predict for new house

    IMPORTANT:
    Apply SAME transformations
    """
    x = x.reshape(1, -1)

    x_poly = polynomial_features(x)
    x_poly = (x_poly - mu) / sigma

    return np.dot(x_poly, w) + b


x_new = np.array([2000, 3, 2, 20])
print("\nNew Prediction:", predict_new(x_new, w_final, b_final, mu, sigma))


# ================================
# STEP 12: VISUALIZATION
# ================================

# Use only SIZE for plotting
X_vis = X_train[:, 0]
y_vis = y_train

# Sort for smooth curve
sorted_idx = np.argsort(X_vis)
X_vis = X_vis[sorted_idx]
y_vis = y_vis[sorted_idx]


def predict_curve(x_vals):
    """
    Predict curve using only size
    other features fixed
    """
    preds = []

    for x in x_vals:
        sample = np.array([x, 3, 2, 30])
        sample = sample.reshape(1, -1)

        sample_poly = polynomial_features(sample)
        sample_poly = (sample_poly - mu) / sigma

        pred = np.dot(sample_poly, w_final) + b_final
        preds.append(pred[0])

    return np.array(preds)


x_range = np.linspace(min(X_vis), max(X_vis), 100)
y_curve = predict_curve(x_range)

# Plot
plt.scatter(X_vis, y_vis, label="Actual Data")
plt.plot(x_range, y_curve, color='red', label="Model Curve")

plt.title("Polynomial Regression (Size vs Price)")
plt.xlabel("Size")
plt.ylabel("Price")
plt.legend()
plt.show()


# ================================
# STEP 13: COST GRAPH
# ================================
plt.plot(J_hist)
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()