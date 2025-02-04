import numpy as np
import matplotlib.pyplot as plt

def mse_loss(actual, predicted):
    """Compute Mean Squared Error."""
    return np.mean((actual - predicted) ** 2)

def gradient_descent(X, Y, lr=0.0001, max_iters=2000, tolerance=1e-6):
    """Perform gradient descent to optimize weight and bias."""
    
    # Initialize parameters
    weight = 0.1
    bias = 0.01
    num_samples = len(X)
    
    prev_loss = float('inf')
    loss_history = []
    weight_history = []

    for i in range(max_iters):
        # Computing predictions
        Y_pred = weight * X + bias
        
        # Computing loss
        loss = mse_loss(Y, Y_pred)
        loss_history.append(loss)
        weight_history.append(weight)
        
        # stopping condition
        if abs(prev_loss - loss) <= tolerance:
            break
        
        prev_loss = loss
        
        # Computing gradients
        weight_grad = (-2 / num_samples) * np.sum(X * (Y - Y_pred))
        bias_grad = (-2 / num_samples) * np.sum(Y - Y_pred)
        
        # Update parameters
        weight -= lr * weight_grad
        bias -= lr * bias_grad
        
        # SHow progress at every 500 iterations
        if (i + 1) % 500 == 0 or i == 0:
            print(f"Iteration {i+1}: Loss={loss:.5f}, Weight={weight:.5f}, Bias={bias:.5f}")

    # Plot loss vs weight
    plt.figure(figsize=(8, 6))
    plt.plot(weight_history, loss_history, marker='o', color='red', linestyle='dashed')
    plt.xlabel("Weight")
    plt.ylabel("Loss")
    plt.title("Loss vs Weight")
    plt.show()
    
    return weight, bias

def main():
    """Main function to execute gradient descent."""
    
    # Sample dataset
    X = np.array([32.5, 53.4, 61.5, 47.4, 59.8, 55.1, 52.2, 39.3, 48.1, 52.5,
                  45.4, 54.3, 44.1, 58.1, 56.7, 48.9, 44.6, 60.2, 45.6, 38.8])
    Y = np.array([31.7, 68.7, 62.5, 71.5, 87.2, 78.2, 79.6, 59.1, 75.3, 71.3,
                  55.1, 82.4, 62.0, 75.3, 81.4, 60.7, 82.8, 97.3, 48.8, 56.8])
    
    # Run gradient descent
    final_weight, final_bias = gradient_descent(X, Y)
    print(f"Optimized Weight: {final_weight:.5f}")
    print(f"Optimized Bias: {final_bias:.5f}")

    # Plot results
    Y_pred = final_weight * X + final_bias

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='red', label="Actual Data")
    plt.plot(X, Y_pred, color='blue', linestyle='dashed', label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Linear Regression with Gradient Descent")
    plt.show()

if __name__ == "__main__":
    main()
