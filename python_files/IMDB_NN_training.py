import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import ast
import warnings
import matplotlib.pyplot as plt
import os
import datetime
import sys
import pickle

warnings.filterwarnings('ignore')

# Create directories for saving graphs and logs
GRAPHS_DIR = "IMDB_NN_training_graphs"
LOGS_DIR = "IMDB_NN_training_logs"
MODEL_DIR = "IMDB_NN_models"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Setup logging to file
log_filename = f"{LOGS_DIR}/NN_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
print(f"=== IMDB Neural Network Binary Sentiment Analysis Training Log ===")
print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_filename}")
print("=" * 70)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets from the specified paths
print("Loading datasets...")
TRAIN_PATH = "/home/m23mac008/NLU/TOKENIZED_Train_IMBD.CSV"
TEST_PATH = "/home/m23mac008/NLU/TOKENIZED_Test_IMBD.CSV"

# Load training and test datasets
print("Loading train dataset...")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Train set size: {len(train_df)}")

print("Loading test dataset...")
test_df = pd.read_csv(TEST_PATH)
print(f"Test set size: {len(test_df)}")

# Display sentiment distribution
print("\nSentiment distribution in train set:")
print(train_df['sentiment'].value_counts())

print("\nSentiment distribution in test set:")
print(test_df['sentiment'].value_counts())

# Convert token_indices from string to actual list
print("Converting token indices to lists...")
train_df['token_indices'] = train_df['token_indices'].apply(ast.literal_eval)
test_df['token_indices'] = test_df['token_indices'].apply(ast.literal_eval)

# Determine vocabulary size and max length
# If you have a vocab pickle file, load it
# Otherwise, calculate these values from the data
try:
    IMDB_DIR = os.path.dirname(TRAIN_PATH)
    vocab_file = os.path.join(IMDB_DIR, "IMDB_vocab.pkl")
    
    if os.path.exists(vocab_file):
        print("Loading vocabulary information...")
        with open(vocab_file, 'rb') as f:
            vocab_info = pickle.load(f)
        vocab_size = vocab_info['vocab_size']
        max_length = vocab_info['max_length']
        print(f"Loaded vocabulary size: {vocab_size}")
        print(f"Loaded maximum sequence length: {max_length}")
    else:
        print("Vocabulary file not found. Calculating from data...")
        # Find all unique token indices
        all_indices = set()
        for indices in train_df['token_indices']:
            all_indices.update(indices)
        vocab_size = max(all_indices) + 1  # +1 because indices are 0-based
        
        # Find max sequence length
        max_length = max(len(indices) for indices in train_df['token_indices'])
        
        # Save vocab info for future use
        vocab_info = {'vocab_size': vocab_size, 'max_length': max_length}
        with open(os.path.join(IMDB_DIR, "IMDB_vocab.pkl"), 'wb') as f:
            pickle.dump(vocab_info, f)
except Exception as e:
    print(f"Error processing vocabulary: {e}")
    print("Calculating vocabulary size and max length from training data...")
    
    # Find all unique token indices
    all_indices = set()
    for indices in train_df['token_indices']:
        all_indices.update(indices)
    vocab_size = max(all_indices) + 1  # +1 because indices are 0-based
    
    # Find max sequence length
    max_length = max(len(indices) for indices in train_df['token_indices'])

print(f"Vocabulary size: {vocab_size}")
print(f"Maximum sequence length: {max_length}")


# Split the training set: use the last 10% as validation set
print("\nSplitting train set: Using last 10% as validation set...")
train_size = int(0.9 * len(train_df))
train_data = train_df.iloc[:train_size]
val_data = train_df.iloc[train_size:]

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# Define the dataset class for Neural Network
class IMDBDataset(Dataset):
    def __init__(self, token_indices, labels, max_length=None):
        self.token_indices = token_indices
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get the token indices for this example
        indices = self.token_indices[idx]
        
        # If max_length is specified, ensure all sequences have the same length
        if self.max_length is not None:
            if len(indices) > self.max_length:
                indices = indices[:self.max_length]  # Truncate
            
            # Create a sparse tensor using indices directly
            # This is much more memory efficient than one-hot encoding
            # We'll use an embedding layer in the model instead
            padded_indices = torch.zeros(self.max_length, dtype=torch.long)
            padded_indices[:len(indices)] = torch.tensor(indices, dtype=torch.long)
            return padded_indices, self.labels[idx]
        else:
            return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# Define the neural network with embedding layer and two hidden layers
class NNBinarySentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden1_size=256, hidden2_size=128, output_size=1, max_length=None):
        super(NNBinarySentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_length = max_length
        
        # Calculate input size for first linear layer
        if max_length is not None:
            self.input_size = max_length * embedding_dim
        else:
            # Will need to handle variable length in forward pass
            self.input_size = None
        
        self.hidden1_size = hidden1_size
        self.layer1 = nn.Linear(self.input_size if self.input_size else embedding_dim, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # If max_length is not provided, we need to handle variable length inputs
        if self.max_length is None:
            # Use mean pooling over sequence length for variable length inputs
            x = torch.mean(x, dim=1)  # Shape: [batch_size, embedding_dim]
        else:
            # Reshape for fixed length inputs
            x = x.view(-1, self.input_size)  # Shape: [batch_size, seq_len * embedding_dim]
        
        # Apply feedforward layers
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Function to plot and save training/validation metrics
def plot_metrics(epochs, train_metrics, val_metrics, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-o', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-o', label=f'Validation {metric_name}')
    
    title = f'IMDB Neural Network Training and Validation {metric_name}'
    filename = f"{GRAPHS_DIR}/nn_{metric_name.lower()}.png"
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Print model architecture and hyperparameters
print("\n=== Neural Network Model Architecture and Hyperparameters ===")
embedding_dim = 100  # Embedding dimension
hidden1_size = 256
hidden2_size = 128
output_size = 1    # For binary classification
learning_rate = 0.001
num_epochs = 10
batch_size = 64

print(f"Dataset: IMDB movie reviews (binary sentiment classification)")
print(f"Model type: Neural Network with Embedding Layer")
print(f"Vocabulary size: {vocab_size}")
print(f"Maximum sequence length: {max_length}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Hidden layer 1 size: {hidden1_size}")
print(f"Hidden layer 2 size: {hidden2_size}")
print(f"Output size: {output_size} (binary classification)")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Activation function: ReLU")
print(f"Loss function: BCEWithLogitsLoss (for binary classification)")
print(f"Optimizer: Adam")
print(f"Using GPU: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print("=" * 70)

# Prepare datasets
train_token_indices = train_data['token_indices'].tolist()
train_labels = train_data['sentiment'].values

val_token_indices = val_data['token_indices'].tolist()
val_labels = val_data['sentiment'].values

test_token_indices = test_df['token_indices'].tolist()
test_labels = test_df['sentiment'].values

train_dataset = IMDBDataset(
    train_token_indices,
    torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1),
    max_length=max_length
)

val_dataset = IMDBDataset(
    val_token_indices,
    torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1),
    max_length=max_length
)

test_dataset = IMDBDataset(
    test_token_indices,
    torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1),
    max_length=max_length
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = NNBinarySentimentClassifier(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden1_size=hidden1_size,
    hidden2_size=hidden2_size,
    output_size=output_size,
    max_length=max_length
).to(device)
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
print("\nNeural Network Training Progress:")
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
    val_preds = []
    val_targets = []
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_preds.extend(predictions.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())
    
    # Calculate validation accuracy
    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()
    val_acc = accuracy_score(val_targets, val_preds) * 100  # Convert to percentage
    avg_val_loss = val_loss / len(val_loader)
    
    # Store metrics for plotting
    epochs_list.append(epoch + 1)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    # Print in table format
    print(f"{epoch+1:^6} | {avg_train_loss:^10.4f} | {train_acc:^10.2f}% | {avg_val_loss:^10.4f} | {val_acc:^10.2f}%")
    
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{MODEL_DIR}/nn_best_model.pt")
        print(f"*** New best model saved with validation accuracy: {val_acc:.2f}% ***")

print("-" * 80)

# Plot and save metrics
plot_metrics(epochs_list, train_losses, val_losses, "Loss")
plot_metrics(epochs_list, train_accuracies, val_accuracies, "Accuracy")

# Final test metrics (using the best model)
print("\nLoading best model for testing...")
model.load_state_dict(torch.load(f"{MODEL_DIR}/nn_best_model.pt"))
model.eval()

# Test set evaluation
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Testing on test set"):
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

print("\nTest Results on Best Model Checkpoint:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print("\nPer-Class Metrics:")
for cls, metrics in class_metrics.items():
    print(f"{cls}: Precision={metrics['precision']:.2f}%, Recall={metrics['recall']:.2f}%, F1={metrics['f1']:.2f}%")

print("\nFile Outputs:")
print(f"- IMDB NN Training logs saved to: {log_filename}")
print(f"- IMDB NN Training graphs saved to: {GRAPHS_DIR}/ directory")
print(f"- IMDB NN Best model saved as: {MODEL_DIR}/nn_best_model.pt")

print(f"\nNeural Network Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Additional information about architecture
print("\n=== Note on Architecture for Neural Network ===")
print("For binary classification (IMDB movie review task):")
print("- Used BCEWithLogitsLoss (Binary Cross Entropy with Logits)")
print("- Embedding layer for efficient token representation")
print("- Feed-forward network with two hidden layers")
print("- First hidden layer: 256 neurons")
print("- Second hidden layer: 128 neurons") 
print("- ReLU activation and dropout for regularization")
print("- Optimized for GPU usage with embedding instead of one-hot encoding")
print("=" * 70)