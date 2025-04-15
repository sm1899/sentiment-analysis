import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import ast
import warnings
import matplotlib.pyplot as plt
import os
import datetime
import sys
warnings.filterwarnings('ignore')

# Create directories for saving graphs and logs with LSTM in the name
GRAPHS_DIR = "SemEval_lstm_training_graphs"
LOGS_DIR = "SemEval_lstm_training_logs"
MODEL_DIR = "SemEval_lstm_models"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Setup logging to file
log_filename = f"{LOGS_DIR}/lstm_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Create a class to log both to console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both console and file
sys.stdout = Logger(log_filename)

# Start logging
print(f"=== SemEval LSTM Binary Sentiment Analysis Training Log ===")
print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_filename}")
print("=" * 70)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
print("Loading dataset...")
df = pd.read_csv("/home/m23mac008/NLU/TOKENIZED_SemEval.CSV")

print("Original sentiment distribution:")
print(df['sentiment'].value_counts())

# Convert token_indices from string to actual list
print("Converting token indices to lists...")
df['token_indices'] = df['token_indices'].apply(ast.literal_eval)

# Map sentiment labels for binary classification (0 -> 0, 4 -> 1)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 0 if x == 0 else 1)
print("\nRemapped sentiment distribution (0=negative, 1=positive):")
print(df['sentiment_label'].value_counts())

# Find the vocabulary size (maximum index in the token_indices + 1)
vocab_size = max(max(indices) for indices in df['token_indices']) + 1
print(f"Vocabulary size: {vocab_size}")

# Find the sequence length (length of token_indices lists)
seq_length = len(df['token_indices'].iloc[0])
print(f"Sequence length: {seq_length}")

# Define the dataset class for LSTM with binary labels
class SemEvalLSTMDataset(Dataset):
    def __init__(self, token_indices, labels):
        self.token_indices = token_indices
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # For LSTM, we need the sequence, not one-hot encoded
        indices = torch.tensor(self.token_indices[idx], dtype=torch.long)
        return indices, self.labels[idx]

# Define the LSTM model for binary classification
class LSTMBinarySentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, dropout=0.3):
        super(LSTMBinarySentimentClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=1, 
                           batch_first=True, 
                           dropout=0 if dropout == 0 else dropout)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch size, seq length]
        
        # Apply embedding layer
        embedded = self.embedding(text)
        # embedded shape: [batch size, seq length, embedding dim]
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden shape: [1, batch size, hidden dim]
        
        # We use the last hidden state for classification as specified by Karpathy
        hidden = hidden.squeeze(0)
        # hidden shape: [batch size, hidden dim]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Dense layer for classification (no activation - handled by BCEWithLogitsLoss)
        output = self.fc(hidden)
        # output shape: [batch size, 1]
        
        return output

# Function to plot and save training/validation metrics
def plot_metrics(epochs, train_metrics, val_metrics, metric_name, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-o', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-o', label=f'Validation {metric_name}')
    plt.title(f'SemEval LSTM Training and Validation {metric_name} - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{GRAPHS_DIR}/lstm_fold_{fold}_{metric_name.lower()}.png")
    plt.close()

# Function to plot average metrics across all folds
def plot_average_metrics(epochs, train_metrics_all_folds, val_metrics_all_folds, metric_name):
    train_avg = np.mean(train_metrics_all_folds, axis=0)
    val_avg = np.mean(val_metrics_all_folds, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_avg, 'b-o', label=f'Avg Training {metric_name}')
    plt.plot(epochs, val_avg, 'r-o', label=f'Avg Validation {metric_name}')
    plt.title(f'SemEval LSTM Average Training and Validation {metric_name} Across All Folds')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{GRAPHS_DIR}/lstm_average_{metric_name.lower()}.png")
    plt.close()

# Print model architecture and hyperparameters
print("\n=== LSTM Model Architecture and Hyperparameters ===")
embedding_dim = 100
hidden_dim = 256  # As specified in instructions
output_dim = 1    # For binary classification
dropout = 0.3
learning_rate = 0.001
num_epochs = 10
batch_size = 64

print(f"Dataset: SemEval binary sentiment classification")
print(f"Model type: LSTM")
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Hidden layer size: {hidden_dim}")
print(f"Output size: {output_dim} (binary classification)")
print(f"Dropout rate: {dropout}")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Loss function: BCEWithLogitsLoss (for binary classification)")
print(f"Optimizer: Adam")
print("=" * 70)

# Lists to store metrics for all folds
all_train_acc = []
all_val_acc = []
all_train_loss = []
all_val_loss = []

# Convert data to appropriate format
token_indices = df['token_indices'].tolist()
labels = df['sentiment_label'].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

print("\n=== Starting 5-Fold Cross-Validation with LSTM ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(token_indices)):
    print(f"\n--- Fold {fold+1}/5 ---")
    print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
    
    # Create train and validation datasets
    train_dataset = SemEvalLSTMDataset(
        [token_indices[i] for i in train_idx],
        torch.tensor([labels[i] for i in train_idx], dtype=torch.float32).unsqueeze(1)  # For binary classification
    )
    
    val_dataset = SemEvalLSTMDataset(
        [token_indices[i] for i in val_idx],
        torch.tensor([labels[i] for i in val_idx], dtype=torch.float32).unsqueeze(1)  # For binary classification
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = LSTMBinarySentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, dropout).to(device)
    
    # Use BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_list = []
    
    # Training loop
    best_val_acc = 0.0
    print("\nLSTM Training Progress:")
    print("-" * 80)
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10}")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Get predictions for accuracy calculation
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(predictions.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training accuracy
        train_preds = np.array(train_preds).flatten()
        train_targets = np.array(train_targets).flatten()
        train_acc = accuracy_score(train_targets, train_preds) * 100  # Convert to percentage
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        all_val_preds = []
        all_val_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation accuracy
        all_val_preds = np.array(all_val_preds).flatten()
        all_val_targets = np.array(all_val_targets).flatten()
        val_acc = accuracy_score(all_val_targets, all_val_preds) * 100  # Convert to percentage
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics for plotting
        epochs_list.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print in table format
        print(f"{epoch+1:^6} | {avg_train_loss:^10.4f} | {train_acc:^10.2f}% | {avg_val_loss:^10.4f} | {val_acc:^10.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_best_model_fold{fold+1}.pt")
            print(f"*** New best model saved with validation accuracy: {val_acc:.2f}% ***")
    
    print("-" * 80)
    
    # Plot and save metrics for this fold
    plot_metrics(epochs_list, train_losses, val_losses, "Loss", fold+1)
    plot_metrics(epochs_list, train_accuracies, val_accuracies, "Accuracy", fold+1)
    
    # Store metrics for average plots
    all_train_acc.append(train_accuracies)
    all_val_acc.append(val_accuracies)
    all_train_loss.append(train_losses)
    all_val_loss.append(val_losses)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_best_model_fold{fold+1}.pt"))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    accuracy = accuracy_score(all_targets, all_preds) * 100  # Convert to percentage
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    # Convert to percentages
    precision *= 100
    recall *= 100
    f1 *= 100
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=[0, 1]
    )
    # Convert to percentages
    per_class_precision *= 100
    per_class_recall *= 100
    per_class_f1 *= 100
    
    class_metrics = {
        'Negative (0)': {'precision': per_class_precision[0], 'recall': per_class_recall[0], 'f1': per_class_f1[0]},
        'Positive (1)': {'precision': per_class_precision[1], 'recall': per_class_recall[1], 'f1': per_class_f1[1]}
    }
    
    fold_results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    })
    
    print(f"\nLSTM Fold {fold+1} Final Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print("\nPer-Class Metrics:")
    for cls, metrics in class_metrics.items():
        print(f"{cls}: Precision={metrics['precision']:.2f}%, Recall={metrics['recall']:.2f}%, F1={metrics['f1']:.2f}%")

# Plot average metrics across all folds
plot_average_metrics(epochs_list, all_train_loss, all_val_loss, "Loss")
plot_average_metrics(epochs_list, all_train_acc, all_val_acc, "Accuracy")

# Calculate and report average metrics across all folds
avg_acc = sum(fold['accuracy'] for fold in fold_results) / len(fold_results)
avg_precision = sum(fold['precision'] for fold in fold_results) / len(fold_results)
avg_recall = sum(fold['recall'] for fold in fold_results) / len(fold_results)
avg_f1 = sum(fold['f1'] for fold in fold_results) / len(fold_results)

avg_class_metrics = {
    'Negative (0)': {
        'precision': sum(fold['class_metrics']['Negative (0)']['precision'] for fold in fold_results) / len(fold_results),
        'recall': sum(fold['class_metrics']['Negative (0)']['recall'] for fold in fold_results) / len(fold_results),
        'f1': sum(fold['class_metrics']['Negative (0)']['f1'] for fold in fold_results) / len(fold_results)
    },
    'Positive (1)': {
        'precision': sum(fold['class_metrics']['Positive (1)']['precision'] for fold in fold_results) / len(fold_results),
        'recall': sum(fold['class_metrics']['Positive (1)']['recall'] for fold in fold_results) / len(fold_results),
        'f1': sum(fold['class_metrics']['Positive (1)']['f1'] for fold in fold_results) / len(fold_results)
    }
}

print("\n" + "=" * 70)
print("===== SemEval LSTM 5-Fold Cross-Validation Results =====")
print("=" * 70)
print(f"Average Accuracy: {avg_acc:.2f}%")
print(f"Average Precision: {avg_precision:.2f}%")
print(f"Average Recall: {avg_recall:.2f}%")
print(f"Average F1 Score: {avg_f1:.2f}%")

print("\nAverage Per-Class Metrics:")
print("-" * 70)
print(f"{'Class':^12} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}")
print("-" * 70)
for cls, metrics in avg_class_metrics.items():
    print(f"{cls:^12} | {metrics['precision']:^10.2f}% | {metrics['recall']:^10.2f}% | {metrics['f1']:^10.2f}%")
print("-" * 70)

print("\nFile Outputs:")
print(f"- SemEval LSTM Training logs saved to: {log_filename}")
print(f"- SemEval LSTM Training graphs saved to: {GRAPHS_DIR}/ directory")
print(f"- SemEval LSTM Best models saved as: {MODEL_DIR}/lstm_best_model_fold[1-5].pt")

print(f"\nLSTM Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Additional information about loss function choices
print("\n=== Note on Loss Functions for LSTM ===")
print("For binary classification (SemEval task):")
print("- Used BCEWithLogitsLoss (Binary Cross Entropy with Logits)")
print("- Combines Sigmoid activation and binary cross entropy in one layer")
print("- More numerically stable than using separate Sigmoid + BCELoss")
print("- Appropriate for 2-class tasks (negative/positive sentiment)")
print("\nAs per Karpathy's RNN architecture recommendations:")
print("- Using last hidden state from LSTM for classification")
print("- LSTM processes the entire sequence and final hidden state is used")
print("- This approach captures the overall sentiment of the text")
print("=" * 70)