"""
Simple Model Retraining Script
Allows clients to retrain pre-trained models with custom datasets
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("Deep learning libraries not available. Only ML models will work.")

# Define CNN1D model
class CNN1D(nn.Module):
    def __init__(self, input_features=12):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (input_features // 2), 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_features=12):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_dataset(file_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(file_path)
    
    # Assuming last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Ensure target is binary (0 or 1)
    y = (y > 0).astype(int)
    
    return X, y

def train_ml_model(model_type, X_train, y_train, X_test, y_test):
    """Train traditional ML model"""
    
    # Initialize model
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_type == 'gradient_boost':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    return model, metrics

def train_dl_model(model_type, X_train, y_train, X_test, y_test, epochs=50):
    """Train deep learning model"""
    
    if not DL_AVAILABLE:
        raise RuntimeError("Deep learning libraries not available")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    input_features = X_train.shape[1]
    if model_type == 'cnn1d':
        model = CNN1D(input_features=input_features)
    elif model_type == 'cnn_lstm':
        model = CNNLSTM(input_features=input_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        y_pred = torch.argmax(probabilities, dim=1).numpy()
        y_pred_proba = probabilities[:, 1].numpy()
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
    }
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Retrain ML/DL models')
    parser.add_argument('--model-type', required=True, 
                       choices=['logistic_regression', 'svm', 'gradient_boost', 'knn', 'cnn1d', 'cnn_lstm'])
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--output-dir', default='models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for DL models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {args.data_path}...")
    X, y = load_dataset(args.data_path)
    
    print(f"Dataset shape: {X.shape}, Labels: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = output_dir / f'{args.model_type}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nTraining {args.model_type} model...")
    
    # Train based on model type
    if args.model_type in ['cnn1d', 'cnn_lstm']:
        model, metrics = train_dl_model(args.model_type, X_train, y_train, X_test, y_test, args.epochs)
        # Save PyTorch model
        model_path = output_dir / f'{args.model_type}_model.pth'
        torch.save(model.state_dict(), model_path)
    else:
        model, metrics = train_ml_model(args.model_type, X_train, y_train, X_test, y_test)
        # Save sklearn model
        model_path = output_dir / f'{args.model_type}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"\nModel saved to {model_path}")
    print("\nTraining Results:")
    print(json.dumps(metrics, indent=2))
    
    # Save metrics to JSON
    metrics_path = output_dir / f'{args.model_type}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'metrics': metrics,
            'data_shape': X.shape[0],
            'features': X.shape[1]
        }, f, indent=2)
    
    # Output success message
    result = {
        'success': True,
        'model_type': args.model_type,
        'metrics': metrics,
        'model_path': str(model_path),
        'scaler_path': str(scaler_path)
    }
    
    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
