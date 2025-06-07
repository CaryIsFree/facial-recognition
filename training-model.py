import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_simple_data(n_samples=100, noise=0.2):
    """
    Generate a simple synthetic dataset with linear relationship and noise.
    
    Args:
        n_samples: Number of data points
        noise: Amount of noise to add
        
    Returns:
        X: Input features of shape (n_samples, 2)
        y: Labels of shape (n_samples, 1)
    """
    # Create two input features
    X = np.random.randn(n_samples, 2)
    
    # Generate target: if x1 + x2 > 0, then 1, else 0 (with some noise)
    y_clean = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    y = y_clean + np.random.randn(n_samples) * noise
    y = (y > 0.5).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    return X_tensor, y_tensor

# Generate our synthetic dataset
X, y = generate_simple_data(n_samples=200)
print(f"Generated data: X shape: {X.shape}, y shape: {y.shape}")

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='coolwarm')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.title("Simple Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(alpha=0.3)
plt.savefig('simple_data.png')
plt.close()

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        """
        A simple neural network for binary classification.
        
        Args:
            input_size: Number of input features (2 in our case)
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes (1 for binary classification)
        """
        super(SimpleClassifier, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Create the model
model = SimpleClassifier()
print(model)

# TODO: Implement the training function
def train_model(model, X, y, n_epochs=100):
    """
    Train the model on the provided data.
    
    Args:
        model: The neural network model
        X: Input features
        y: Target labels
        n_epochs: Number of training epochs
        
    Returns:
        losses: List of loss values during training
    """
    
    #Initalized the loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Store losses for plotting
    losses = []
    
    # Training loop
    for epoch in range(n_epochs):
        #Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X)

        # Compute loss
        loss = loss_fn(outputs, y)

        # Backward pass and optimization
        loss.backward()

        #Update the weights
        optimizer.step()
        
        # Store and print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")  # Update this with actual loss
            losses.append(loss)  # Update this with actual loss
    
    return losses

# Function to make predictions
def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained neural network model
        X: Input features
        
    Returns:
        predictions: Binary predictions (0 or 1)
    """
    model.eval()
    with torch.no_grad():
        # Get predicted probabilities
        outputs = model(X)
        # Convert to binary predictions
        predictions = (torch.sigmoid(outputs) > 0.5).float()
    return predictions

# Visualize decision boundary
def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary of the model.
    
    Args:
        model: Trained model
        X: Input features
        y: True labels
    """
    # Define the grid
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on the grid
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(grid)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    # Reshape for plotting
    predictions = predictions.view(xx.shape).numpy()
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, cmap='coolwarm', alpha=0.3)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), cmap='coolwarm', edgecolor='k', s=50)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(alpha=0.3)
    plt.savefig('decision_boundary.png')
    plt.close()

if __name__ == "__main__":
    # Train the model
    losses = train_model(model, X, y)
    #Convert the losses to as a different array for plotting
    fi_los = [fl.item() for fl in losses]
    
    # Plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, len(fi_los) * 10, 10), fi_los, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Evaluate the model
    predictions = predict(model, X)
    accuracy = (predictions == y).float().mean().item()
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Plot decision boundary
    plot_decision_boundary(model, X, y)
    
    print("Training completed! Check the generated images to see the results.")
