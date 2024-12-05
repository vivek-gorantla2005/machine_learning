import numpy as np

def mean_squared_error(X, y, weights, b):
    n = len(y)
    predictions = np.dot(X, weights) + b
    mse = np.mean((y - predictions) ** 2)
    return mse

def gradient_descent(X, y, weights, b, L):
    n = len(y)
    predictions = np.dot(X, weights) + b
    
    weights_gradient = -(2 / n) * np.dot(X.T, (y - predictions))
    b_gradient = -(2 / n) * np.sum(y - predictions)
    
    weights -= L * weights_gradient
    b -= L * b_gradient
    
    return weights, b

def main():
    dataset = np.array([
        [1, 2, 3, 6],
        [2, 4, 6, 12],
        [3, 6, 9, 18],
        [4, 8, 12, 24],
        [5, 10, 15, 30],
        [6, 12, 18, 36],
        [7, 14, 21, 42],
        [8, 16, 24, 48],
        [9, 18, 27, 54],
        [10, 20, 30, 60],
    ])
    
    X = dataset[:, :-1]
    y = dataset[:, -1]

    # Normalize the features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    weights = np.zeros(X.shape[1])
    b = 0
    L = 0.001  # Lower learning rate
    epochs = 1000

    print("Starting gradient descent...\n")

    for epoch in range(epochs):
        weights, b = gradient_descent(X, y, weights, b, L)
        if epoch % 100 == 0:
            mse = mean_squared_error(X, y, weights, b)
            print(f"Epoch {epoch}, MSE: {mse:.4f}, Weights: {weights}, b: {b:.4f}")

    print("\nFinal parameters:")
    print(f"Weights: {weights}")
    print(f"Intercept (b): {b:.4f}")

    predictions = np.dot(X, weights) + b
    print("\nPredictions:")
    print(predictions)

if __name__ == "__main__":
    main()
