import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    def __init__(self, condition_size, feature_size):
        super(FiLMLayer, self).__init__()
        # FiLM generates gamma and beta for feature-wise modulation
        self.gamma_net = nn.Linear(condition_size, feature_size)
        self.beta_net = nn.Linear(condition_size, feature_size)
        
    def forward(self, x, condition):
        # x: (batch_size, feature_size)
        # condition: (batch_size, condition_size) - one-hot encoded vector
        gamma = self.gamma_net(condition)  # (batch_size, feature_size)
        beta = self.beta_net(condition)    # (batch_size, feature_size)
        
        # Apply feature-wise linear modulation: gamma * x + beta
        return gamma * x + beta

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # Input: features 1-22
        self.input_size = 22
        # Hidden layer: 120 nodes, each connected to 2 features
        self.hidden_size = 120
        # Output: placeholder for now
        self.output_size = 1  # This can be changed as needed

        # Define the connections for the first hidden layer
        # Each node connects to one feature from 1-6 or 19-22, and one from 7-18
        self.group1_indices = list(range(0, 6)) + list(range(18, 22))  # 1-6, 19-22 (0-based), location features
        self.group2_indices = list(range(6, 18))  # 7-18 (0-based), number features

        # Precompute all possible pairs
        self.pairs = []
        for i in self.group1_indices:
            for j in self.group2_indices:
                self.pairs.append((i, j))

        # Each hidden node has 2 weights and a bias
        self.hidden_weights = nn.Parameter(torch.randn(self.hidden_size, 2))
        self.hidden_bias = nn.Parameter(torch.zeros(self.hidden_size))

        # FiLM layer for modulating hidden layer with one-hot encoded condition
        self.film = FiLMLayer(condition_size=3, feature_size=self.hidden_size)

        # Output layer (for now, just a placeholder)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # x: (batch_size, 22)
        # Split input into features and condition
        features = x[:, :22]  # features 0-21
        condition = x[:, 22:]  # features 22-24 - one-hot encoded
        
        # Build the hidden layer activations
        h = []
        for idx, (i, j) in enumerate(self.pairs):
            # For each hidden node, get its two input features
            xi = features[:, i]
            xj = features[:, j]
            # Weighted sum + bias
            hi = self.hidden_weights[idx, 0] * xi + self.hidden_weights[idx, 1] * xj + self.hidden_bias[idx]
            h.append(hi)
        h = torch.stack(h, dim=1)  # (batch_size, hidden_size)
        
        # Apply FiLM modulation
        h = self.film(h, condition)
        
        # Apply activation function
        h = F.relu(h)
        
        # Output layer
        out = self.output(h)
        return out

def train_feature_net(model, train_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    """
    Training function for FeatureNet
    
    Args:
        model: FeatureNet model
        train_loader: DataLoader containing training data (features, labels)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cpu' or 'cuda')
    """
    model = model.to(device)
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Backward pass (automatic in PyTorch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
    
    print("Training completed!")
    return model

def evaluate_feature_net(model, test_loader, device='cpu'):
    """
    Evaluation function for FeatureNet
    
    Args:
        model: Trained FeatureNet model
        test_loader: DataLoader containing test data (features, labels)
        device: Device to evaluate on ('cpu' or 'cuda')
    
    Returns:
        Average loss on test set
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Test Loss: {avg_loss:.6f}")
    return avg_loss

def create_data_loader(features, labels, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader from features and labels
    
    Args:
        features: List of feature vectors
        labels: List of corresponding labels
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
    
    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

if __name__ == "__main__":
    # Example usage
    net = FeatureNet()
    dummy_input = torch.randn(4, 22)  # batch of 4 samples
    output = net(dummy_input)
    print("Output shape:", output.shape)
    
    # Example training setup
    print("\nExample training setup:")
    print("1. Prepare your features and labels")
    print("2. Create DataLoader: loader = create_data_loader(features, labels)")
    print("3. Train the model: train_feature_net(net, loader)")
    print("4. Evaluate: evaluate_feature_net(net, test_loader)") 