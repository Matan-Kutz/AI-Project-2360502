#!/usr/bin/env python3

import torch
import torch.nn as nn
from Feature_net import FeatureNet, train_feature_net, evaluate_feature_net, create_data_loader
from GameFeatures import GameFeatures
import os
import random
import numpy as np

def prepare_training_data_from_directory(game_features, log_directory):
    """
    Prepare training data from all log files in a directory
    
    Args:
        game_features: GameFeatures object
        log_directory: Path to the directory containing log files
    
    Returns:
        features: List of feature vectors
        labels: List of labels (1 for winner moves, 0 for loser moves)
    """
    features = []
    labels = []
    
    # Find all log files in the directory
    log_files = []
    log_number = 0
    
    while True:
        log_filename = f"log {log_number}"
        log_path = os.path.join(log_directory, log_filename)
        
        if os.path.exists(log_path):
            log_files.append(log_path)
            log_number += 1
        else:
            break
    
    if not log_files:
        print(f"No log files found in directory: {log_directory}")
        return features, labels
    
    print(f"Found {len(log_files)} log files: {log_files}")
    
    # Process each log file
    for log_file in log_files:
        try:
            print(f"Processing {log_file}...")
            
            # Read game transcript
            moves_dict = game_features.read_game_transcript(log_file)
            winner_moves = moves_dict['winner_moves']
            loser_moves = moves_dict['loser_moves']
            
            # Create feature lists
            feature_dict = game_features.create_feature_lists(moves_dict)
            winner_features = feature_dict['winner_features']
            loser_features = feature_dict['loser_features']
            
            # Add winner moves with label 1
            for feature_list in winner_features:
                features.append(feature_list)
                labels.append(1.0)  # Winner moves get label 1
            
            # Add loser moves with label 0
            for feature_list in loser_features:
                features.append(feature_list)
                labels.append(0.0)  # Loser moves get label 0
            
            print(f"  Added {len(winner_features)} winner moves and {len(loser_features)} loser moves")
            
        except Exception as e:
            print(f"  Error processing {log_file}: {e}")
            continue
    
    return features, labels

def split_data(features, labels, train_ratio=0.8):
    """
    Split data into training and testing sets
    
    Args:
        features: List of feature vectors
        labels: List of labels
        train_ratio: Ratio of data to use for training
    
    Returns:
        train_features, train_labels, test_features, test_labels
    """
    # Combine features and labels
    data = list(zip(features, labels))
    random.shuffle(data)
    
    # Split into train and test
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Separate features and labels
    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)
    
    return list(train_features), list(train_labels), list(test_features), list(test_labels)

def train_feature_net_main():
    print("FeatureNet Training System")
    print("=" * 40)
    
    # Get log directory from user or use default
    import sys
    if len(sys.argv) > 1:
        log_directory = sys.argv[1]
    else:
        log_directory = "."  # Current directory
    
    # Check if directory exists
    if not os.path.exists(log_directory):
        print(f"Error: Directory '{log_directory}' not found.")
        print("Usage: python train_feature_net.py [log_directory]")
        print("If no directory is specified, the current directory will be used.")
        return
    
    try:
        # Create game features object
        print("Creating GameFeatures object...")
        game_features = GameFeatures()
        
        # Prepare training data from all log files in directory
        print(f"Preparing training data from directory: {log_directory}")
        features, labels = prepare_training_data_from_directory(game_features, log_directory)
        
        if not features:
            print("No features found in the transcript.")
            return
        
        print(f"Total samples: {len(features)}")
        print(f"Winner moves: {sum(1 for label in labels if label == 1.0)}")
        print(f"Loser moves: {sum(1 for label in labels if label == 0.0)}")
        print()
        
        # Split data into train and test sets
        train_features, train_labels, test_features, test_labels = split_data(features, labels)
        
        print(f"Training samples: {len(train_features)}")
        print(f"Testing samples: {len(test_features)}")
        print()
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader = create_data_loader(train_features, train_labels, batch_size=16)
        test_loader = create_data_loader(test_features, test_labels, batch_size=16, shuffle=False)
        
        # Create and train the model
        print("Creating FeatureNet model...")
        model = FeatureNet()
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print()
        
        # Train the model
        print("Starting training...")
        trained_model = train_feature_net(
            model=model,
            train_loader=train_loader,
            num_epochs=50,
            learning_rate=0.001,
            device=device
        )
        
        # Evaluate the model
        print("\nEvaluating model...")
        test_loss = evaluate_feature_net(trained_model, test_loader, device)
        
        # Save the trained model
        torch.save(trained_model.state_dict(), 'feature_net_model.pth')
        print("\nModel saved as 'feature_net_model.pth'")
        
        # Example prediction
        print("\nExample prediction:")
        trained_model.eval()
        with torch.no_grad():
            # Use first test sample
            sample_features = torch.tensor([test_features[0]], dtype=torch.float32).to(device)
            prediction = trained_model(sample_features)
            actual_label = test_labels[0]
            print(f"Sample features: {test_features[0]}")
            print(f"Predicted value: {prediction.item():.4f}")
            print(f"Actual label: {actual_label}")
            print(f"Prediction is {'winner' if prediction.item() > 0.5 else 'loser'}")
            print(f"Actual is {'winner' if actual_label > 0.5 else 'loser'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def create_sample_logs():
    """
    Create sample log files for testing
    """
    import os
    
    # Create sample log directory
    log_dir = "sample_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Sample log content
    sample_log_content = """winner:1
turn:1
pos0-0:0:S
pos1-0:1:M
pos2-0:
pos0-1:
pos1-1:
pos2-1:
pos0-2:
pos1-2:
pos2-2:
turn:2
pos0-0:0:S
pos1-0:1:M
pos2-0:0:L
pos0-1:
pos1-1:
pos2-1:
pos0-2:
pos1-2:
pos2-2:"""
    
    # Create multiple sample log files
    for i in range(5):
        log_filename = f"log {i}"
        log_path = os.path.join(log_dir, log_filename)
        with open(log_path, 'w') as f:
            f.write(sample_log_content)
    
    print(f"Created sample log files in directory: {log_dir}")
    print("You can now run: python train_feature_net.py sample_logs")

if __name__ == "__main__":
    # Check if user wants to create sample logs
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-samples":
        create_sample_logs()
    else:
        train_feature_net_main() 