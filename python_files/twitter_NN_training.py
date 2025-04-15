import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
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

# Create directories for saving graphs and logs
GRAPHS_DIR = "twitter_training_graphs"
LOGS_DIR = "twitter_training_logs"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging to file
log_filename = f"{LOGS_DIR}/training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
print(f"=== Sentiment Analysis Training Log ===")
print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_filename}")
print("=" * 50)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
print("Loading dataset...")
df = pd.read_csv("/home/m23mac008/NLU/TOKENIZED_TWITTER.CSV")

print("Sentiment distribution:")
print(df['sentiment'].value_counts())

# Convert token_indices from string to actual list
print("Converting token indices to lists...")
df['token_indices'] = df['token_indices'].apply(ast.literal_eval)

# Map sentiment labels to numerical values
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)

# Find the vocabulary size (maximum index in the token_indices + 1)
vocab_size = max(max(indices) for indices in df['token_indices']) + 1
print(f"Vocabulary size: {vocab_size}")

# Find the sequence length (length of token_indices lists)
seq_length = len(df['token_indices'].iloc[0])
print(f"Sequence length: {seq_length}")

# Define the dataset class
class TwitterSentimentDataset(Dataset):
    def __init__(self, token_indices, labels):
        self.token_indices = token_indices
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Convert token indices to one-hot encoding
        indices = self.token_indices[idx]
        
        # Create a one-hot encoded tensor
        one_hot = torch.zeros(len(indices), vocab_size)
        for i, idx_val in enumerate(indices):
            one_hot[i, idx_val] = 1
        
        # Flatten the one-hot tensor
        flattened = one_hot.view(-1)
        
        return flattened, self.labels[idx]

# Define the neural network with two hidden layers
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden1_size=256, hidden2_size=128, output_size=3):
        super(SentimentClassifier, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Function to plot and save training/validation metrics
def plot_metrics(epochs, train_metrics, val_metrics, metric_name, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-o', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-o', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name} - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{GRAPHS_DIR}/fold_{fold}_{metric_name.lower()}.png")
    plt.close()

# Function to plot average metrics across all folds
def plot_average_metrics(epochs, train_metrics_all_folds, val_metrics_all_folds, metric_name):
    train_avg = np.mean(train_metrics_all_folds, axis=0)
    val_avg = np.mean(val_metrics_all_folds, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_avg, 'b-o', label=f'Avg Training {metric_name}')
    plt.plot(epochs, val_avg, 'r-o', label=f'Avg Validation {metric_name}')
    plt.title(f'Average Training and Validation {metric_name} Across All Folds')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{GRAPHS_DIR}/average_{metric_name.lower()}.png")
    plt.close()

# Print model architecture and hyperparameters
print("\n=== Model Architecture and Hyperparameters ===")
input_size = seq_length * vocab_size
hidden1_size = 256
hidden2_size = 128
output_size = 3 
learning_rate = 0.001
num_epochs = 10
batch_size = 64

print(f"Input size: {input_size}")
print(f"Hidden layer 1 size: {hidden1_size}")
print(f"Hidden layer 2 size: {hidden2_size}")
print(f"Output size: {output_size}")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Activation function: ReLU")
print(f"Loss function: CrossEntropyLoss (for multi-class classification)")
print(f"Optimizer: Adam")
print("=" * 50)

# Convert data to appropriate format
token_indices = df['token_indices'].tolist()
labels = df['sentiment_label'].values

# Lists to store metrics for all folds
all_train_acc = []
all_val_acc = []
all_train_loss = []
all_val_loss = []

# Define model directory
MODEL_DIR = "twitter_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

print("\n=== Starting 5-Fold Cross-Validation ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(token_indices)):
    print(f"\n--- Fold {fold+1}/5 ---")
    print(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
    
    # Create train and validation datasets
    train_dataset = TwitterSentimentDataset(
        [token_indices[i] for i in train_idx],
        torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    )
    
    val_dataset = TwitterSentimentDataset(
        [token_indices[i] for i in val_idx],
        torch.tensor([labels[i] for i in val_idx], dtype=torch.long)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = SentimentClassifier(input_size, hidden1_size, hidden2_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()  # Multi-class loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    epochs_list = []
    
    # Training loop
    best_val_acc = 0.0
    print("\nTraining Progress:")
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
            _, predictions = torch.max(outputs, 1)
            train_preds.extend(predictions.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training accuracy
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
                
                _, predictions = torch.max(outputs, 1)
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation accuracy
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
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model_fold{fold+1}.pt")
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
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model_fold{fold+1}.pt"))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds) * 100  # Convert to percentage
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    # Convert to percentages
    precision *= 100
    recall *= 100
    f1 *= 100
    
    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=[0, 1, 2]
    )
    # Convert to percentages
    per_class_precision *= 100
    per_class_recall *= 100
    per_class_f1 *= 100
    
    class_metrics = {
        'Negative': {'precision': per_class_precision[0], 'recall': per_class_recall[0], 'f1': per_class_f1[0]},
        'Neutral': {'precision': per_class_precision[1], 'recall': per_class_recall[1], 'f1': per_class_f1[1]},
        'Positive': {'precision': per_class_precision[2], 'recall': per_class_recall[2], 'f1': per_class_f1[2]}
    }
    
    fold_results.append({
        'fold': fold + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    })
    
    print(f"\nFold {fold+1} Final Results:")
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
    'Negative': {
        'precision': sum(fold['class_metrics']['Negative']['precision'] for fold in fold_results) / len(fold_results),
        'recall': sum(fold['class_metrics']['Negative']['recall'] for fold in fold_results) / len(fold_results),
        'f1': sum(fold['class_metrics']['Negative']['f1'] for fold in fold_results) / len(fold_results)
    },
    'Neutral': {
        'precision': sum(fold['class_metrics']['Neutral']['precision'] for fold in fold_results) / len(fold_results),
        'recall': sum(fold['class_metrics']['Neutral']['recall'] for fold in fold_results) / len(fold_results),
        'f1': sum(fold['class_metrics']['Neutral']['f1'] for fold in fold_results) / len(fold_results)
    },
    'Positive': {
        'precision': sum(fold['class_metrics']['Positive']['precision'] for fold in fold_results) / len(fold_results),
        'recall': sum(fold['class_metrics']['Positive']['recall'] for fold in fold_results) / len(fold_results),
        'f1': sum(fold['class_metrics']['Positive']['f1'] for fold in fold_results) / len(fold_results)
    }
}

print("\n" + "=" * 50)
print("===== 5-Fold Cross-Validation Results =====")
print("=" * 50)
print(f"Average Accuracy: {avg_acc:.2f}%")
print(f"Average Precision: {avg_precision:.2f}%")
print(f"Average Recall: {avg_recall:.2f}%")
print(f"Average F1 Score: {avg_f1:.2f}%")

print("\nAverage Per-Class Metrics:")
print("-" * 50)
print(f"{'Class':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}")
print("-" * 50)
for cls, metrics in avg_class_metrics.items():
    print(f"{cls:^10} | {metrics['precision']:^10.2f}% | {metrics['recall']:^10.2f}% | {metrics['f1']:^10.2f}%")
print("-" * 50)

print("\nFile Outputs:")
print(f"- Training logs saved to: {log_filename}")
print(f"- Training graphs saved to: {GRAPHS_DIR}/ directory")
print(f"- Best models saved as: {MODEL_DIR}/best_model_fold[1-5].pt")

print(f"\nTraining completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)