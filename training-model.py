import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

#Use the gpu to train model instead of cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#Build the dataset
class LandmarkDataset(Dataset):
    def __init__(self, pkl_file, emotion_to_idx):
        with open(pkl_file, 'rb') as file:
            self.data = pickle.load(file)
        self.emotion_to_idx = emotion_to_idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        
        landmarks = [coord for point in sample['landmarks'] for coord in point]
        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        label = torch.tensor(self.emotion_to_idx[sample['emotion']], dtype=torch.long)

        return landmarks, label

#Specifying the neural network
class Classifier(nn.Module):
    def __init__(self, input_size=1434, hidden_size=128, output_size=7):
        """
        Args:
            input_size: Number of input features, 468 landmarks * 3 coords = 1404
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes, 7 for 7 different emotions
        """
        super(Classifier, self).__init__()
        
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
model = Classifier().to(device)
print(model)

#Load the dataset
emotion_to_idx = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

dataset = LandmarkDataset("facial_landmarks_data.pkl", emotion_to_idx)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(model, dataloader, n_epochs=100):
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Store losses for plotting
    losses = []
    
    # Training loop
    for epoch in range(n_epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)

            # Compute loss
            loss = loss_fn(outputs, y_batch)

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
        predictions
    """
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        # Get predicted probabilities
        outputs = model(X)
        # Convert to class predictions using argmax
        predictions = torch.argmax(outputs, dim=1)
    return predictions.cpu()

if __name__ == "__main__":
    losses = train_model(model, dataloader)

    # Evaluate on full dataset
    all_preds = []
    all_labels = []

    for X_batch, y_batch in dataloader:
        preds = predict(model, X_batch)
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Training accuracy: {accuracy:.4f}")

    # Loss plot
    fi_los = [fl for fl in losses]
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, len(fi_los) * 10, 10), fi_los, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(alpha=0.3)
    plt.savefig('training_loss.png')
    plt.close()
    
    print("Training completed! Check the generated images to see the results.")

    #save the model
    torch.save(model.state_dict(), "emotion_model.pt")
