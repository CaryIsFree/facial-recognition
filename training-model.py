import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader, random_split

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
    def __init__(self, input_size=1434, hidden_size1=128, hidden_size2=64, output_size=7, dropout_prob=0.4):
        """
        Args:
            input_size: Number of input features, 468 landmarks * 3 coords = 1404
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes, 7 for 7 different emotions
        """
        super(Classifier, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        return x

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
#80% train dataset, 20% validation dataset
train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def train_model(model, train_loader, val_loader, n_epochs=150, lr=0.001, wd=1e-5, patience_lr=10, patience_early_stop=20):
    """
    Train the model on the provided data.
    
    Args:
        model: The neural network model
        train_loader
        val_loader
        n_epochs: Number of training epochs, 150
        
    Returns:
        Train Losses
        Val Losses
    """
    
    #Initalized the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_lr)

    # Store losses for plotting
    train_losses, val_losses = [], []

    best_val = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
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
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                total_val_loss += loss_fn(val_outputs, y_val).item()
        
        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}, LR={current_lr:.6f}")
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch+1}: New best validation loss: {best_val:.4f}. Model state saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement in val loss for {patience_early_stop} epochs.")
            if best_model_state:
                model.load_state_dict(best_model_state)
                print("Loaded best model state due to early stopping.")
            break
            
    if best_model_state and (epochs_no_improve < patience_early_stop): # If loop finished before early stopping but a best model was found
        model.load_state_dict(best_model_state)
        print("Loaded best model state from training (loop finished).")
    elif not best_model_state and n_epochs > 0:
        print("Warning: Validation loss did not improve. Model from last epoch is used.")

        # Store and print the loss every 10 epochs
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")
    
    return train_losses, val_losses

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
    # Create the model
    model = Classifier(dropout_prob=0.4).to(device)
    print(model)
    

    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader,
        n_epochs=150,
        lr=0.001,
        wd=1e-5,
        patience_lr=7,
        patience_early_stop=15
        )

    # Evaluate on full dataset
    all_preds = []
    all_labels = []

    for X_batch, y_batch in val_loader:
        preds = predict(model, X_batch)
        all_preds.append(preds.cpu())
        all_labels.append(y_batch.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()
    print(f"Validation accuracy: {accuracy:.4f}")

    # Loss plots
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss & Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('losses.png')
    plt.close()
    
    print("Training completed! Check the generated images to see the results.")

    #save the model
    torch.save(model.state_dict(), "emotion_model.pt")
