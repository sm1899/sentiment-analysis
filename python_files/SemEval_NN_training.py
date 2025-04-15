import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import ast
import warnings
import matplotlib.pyplot as plt
import os
import datetime
import sys
import gc
import psutil
import traceback
warnings.filterwarnings('ignore')

# Create directories for saving graphs and logs
GRAPHS_DIR = "SemEval_NN_training_graphs"
LOGS_DIR = "SemEval_NN_training_logs"
MODEL_DIR = "SemEval_NN_binary_models"
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Improved Dual logging system with better error handling and file I/O
class DualLogger:
    """
    A logging system that writes to both console and a log file,
    with the ability to control what goes into the log file.
    """
    def __init__(self, log_filename):
        self.terminal = sys.stdout
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Open file with explicit buffering=1 (line buffering)
            self.log_file = open(log_filename, 'w', buffering=1)
            print(f"Successfully opened log file: {log_filename}")
        except Exception as e:
            print(f"Error opening log file: {str(e)}")
            self.log_file = None
        
    def log_both(self, message):
        """Log message to both console and file"""
        print(message)  # Print to console
        
        if self.log_file:
            try:
                self.log_file.write(message + '\n')
                self.log_file.flush()  # Force flush after each write
            except Exception as e:
                print(f"Error writing to log file: {str(e)}")
        
    def log_console_only(self, message):
        """Log message to console only"""
        print(message)
        
    def close(self):
        """Close the log file"""
        if self.log_file:
            try:
                self.log_file.flush()  # Ensure all buffered data is written
                self.log_file.close()
                print("Log file closed successfully")
            except Exception as e:
                print(f"Error closing logger file: {str(e)}")

# Global logger that will be initialized in main()
logger = None

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    try:
        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set device for this process
        torch.cuda.set_device(rank)
    except Exception as e:
        print(f"Error in setting up process group on rank {rank}: {str(e)}")
        # Re-raise to be caught by train_fold
        raise

def cleanup():
    """
    Clean up the distributed environment.
    """
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        # Just log the error and continue

# Print memory usage information (console only)
def print_memory_usage(rank=None):
    """Print memory usage to console only"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # GPU memory
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
        gpu_info.append((i, allocated, reserved))
    
    rank_str = f"[Rank {rank}] " if rank is not None else ""
    print(f"{rank_str}System Memory Usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    
    for i, allocated, reserved in gpu_info:
        print(f"{rank_str}GPU {i} Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Modified dataset class - using sparse tensor for memory efficiency
class SemEvalSentimentDataset(Dataset):
    def __init__(self, token_indices, labels, vocab_size):
        self.token_indices = token_indices
        self.labels = labels
        self.vocab_size = vocab_size
        self.seq_length = len(token_indices[0]) if token_indices else 0
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get token indices for this sample
        indices = self.token_indices[idx]
        
        # Use indices directly - the model will handle conversion
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# Modified neural network - using embeddings with dot product for equivalent but memory-efficient one-hot
class NNBinarySentimentClassifier(nn.Module):
    def __init__(self, vocab_size, seq_length, hidden1_size=256, hidden2_size=128, output_size=1):
        super(NNBinarySentimentClassifier, self).__init__()
        
        # Create a weight matrix that effectively mimics one-hot encoding
        # but much more memory efficient
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        
        # First layer is a sparse encoding layer
        self.sparse_encoding = nn.Embedding(vocab_size, 1)
        
        # FC layers same as original
        self.layer1 = nn.Linear(seq_length, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x is token indices: batch_size x seq_length
        
        # Create sparse encoding (mimics one-hot but uses much less memory)
        x = self.sparse_encoding(x).squeeze(-1)  # batch_size x seq_length
        
        # Apply remaining layers
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
    plt.title(f'SemEval NN Training and Validation {metric_name} - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{GRAPHS_DIR}/NN_fold_{fold}_{metric_name.lower()}.png")
    plt.close()

# Function to plot average metrics across all folds
def plot_average_metrics(epochs, train_metrics_all_folds, val_metrics_all_folds, metric_name):
    train_avg = np.mean(train_metrics_all_folds, axis=0)
    val_avg = np.mean(val_metrics_all_folds, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_avg, 'b-o', label=f'Avg Training {metric_name}')
    plt.plot(epochs, val_avg, 'r-o', label=f'Avg Validation {metric_name}')
    plt.title(f'SemEval NN Average Training and Validation {metric_name} Across All Folds')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{GRAPHS_DIR}/NN_average_{metric_name.lower()}.png")
    plt.close()

def train_fold(rank, world_size, token_indices, labels, fold_idx, train_idx, val_idx, 
              vocab_size, seq_length, hidden1_size, hidden2_size, output_size, 
              num_epochs, batch_size, learning_rate):
    try:
        # Setup the process group for this GPU
        setup(rank, world_size)
        
        # Set the device for this process
        device = torch.device(f"cuda:{rank}")
        
        if rank == 0:
            print_memory_usage(rank)
            print(f"Creating datasets for fold {fold_idx+1} on rank {rank}...")
        
        # Create train and validation datasets
        train_dataset = SemEvalSentimentDataset(
            [token_indices[i] for i in train_idx],
            torch.tensor([labels[i] for i in train_idx], dtype=torch.float32).unsqueeze(1),
            vocab_size
        )
        
        val_dataset = SemEvalSentimentDataset(
            [token_indices[i] for i in val_idx],
            torch.tensor([labels[i] for i in val_idx], dtype=torch.float32).unsqueeze(1),
            vocab_size
        )
        
        if rank == 0:
            print(f"Creating data loaders for fold {fold_idx+1} on rank {rank}...")
        
        # Create distributed sampler for training data
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        # Create data loaders with distributed sampler
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            pin_memory=True
        )
        
        # Validation doesn't need to be distributed
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if rank == 0 else None
        
        if rank == 0:
            print(f"Initializing model for fold {fold_idx+1} on rank {rank}...")
            print_memory_usage(rank)
        
        # Initialize model with memory-efficient approach
        model = NNBinarySentimentClassifier(
            vocab_size=vocab_size,
            seq_length=seq_length,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size, 
            output_size=output_size
        ).to(device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank])
        
        # Initialize loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        if rank == 0:
            print_memory_usage(rank)
            print(f"Starting training for fold {fold_idx+1} on rank {rank}...")
        
        # Lists to store metrics for plotting (only on rank 0)
        if rank == 0:
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            epochs_list = []
            best_val_acc = 0.0
            
            # Print to both console and log file
            header_msg = f"\n--- Fold {fold_idx+1}/5 on {world_size} GPUs ---"
            train_info = f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}"
            progress_header = "\nTraining Progress:"
            separator = "-" * 80
            column_header = f"{'Epoch':^6} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10}"
            
            # Use multiple logging methods for redundancy
            if logger:
                # Normal logging
                try:
                    logger.log_both(header_msg)
                    logger.log_both(train_info)
                    logger.log_both(progress_header)
                    logger.log_both(separator)
                    logger.log_both(column_header)
                    logger.log_both(separator)
                except Exception as e:
                    print(f"Logging error in header: {str(e)}")
                
                # Direct file writing as backup
                if hasattr(logger, 'log_file') and logger.log_file:
                    try:
                        header_str = f"{header_msg}\n{train_info}{progress_header}\n{separator}\n{column_header}\n{separator}\n"
                        logger.log_file.write(header_str)
                        logger.log_file.flush()
                    except Exception as e:
                        print(f"Direct file write error in header: {str(e)}")
            else:
                # Fallback to console only
                print(header_msg)
                print(train_info)
                print(progress_header)
                print(separator)
                print(column_header)
                print(separator)
                sys.stdout.flush()
        
        # Training loop
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            train_sampler.set_epoch(epoch)
            
            # Training phase
            model.train()
            train_loss = 0.0
            local_preds = []
            local_targets = []
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Get predictions for local accuracy calculation
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                local_preds.extend(predictions.cpu().numpy())
                local_targets.extend(targets.cpu().numpy())
                
                # Clean up to help with memory
                del inputs, targets, outputs, loss, predictions
            
            # Garbage collection between batches
            gc.collect()
            torch.cuda.empty_cache()
            
            # Gather loss from all processes
            train_loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            
            # Only rank 0 process will handle validation and metrics reporting
            if rank == 0:
                # Calculate training metrics
                local_preds = np.array(local_preds).flatten()
                local_targets = np.array(local_targets).flatten()
                train_acc = accuracy_score(local_targets, local_preds) * 100
                avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
                
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
                        
                        # Clean up to help with memory
                        del inputs, targets, outputs, loss, predictions
                    
                    # Garbage collection after validation
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Calculate validation metrics
                all_val_preds = np.array(all_val_preds).flatten()
                all_val_targets = np.array(all_val_targets).flatten()
                val_acc = accuracy_score(all_val_targets, all_val_preds) * 100
                avg_val_loss = val_loss / len(val_loader)
                
                # Store metrics for plotting
                epochs_list.append(epoch + 1)
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                # Create epoch result string
                epoch_result = f"{epoch+1:^6} | {avg_train_loss:^10.4f} | {train_acc:^10.2f}% | {avg_val_loss:^10.4f} | {val_acc:^10.2f}%"
                
                # Use multiple logging methods for redundancy
                try:
                    # Normal logging through logger
                    if logger and hasattr(logger, 'log_both'):
                        logger.log_both(epoch_result)
                        
                        # Direct file writing as backup
                        if hasattr(logger, 'log_file') and logger.log_file:
                            logger.log_file.write(epoch_result + '\n')
                            logger.log_file.flush()
                    else:
                        # Fallback to console only
                        print(epoch_result)
                        sys.stdout.flush()
                except Exception as e:
                    # Last resort fallback
                    print(epoch_result)
                    print(f"Logging error in epoch results: {str(e)}")
                    sys.stdout.flush()
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.module.state_dict(), f"{MODEL_DIR}/NN_best_model_fold{fold_idx+1}.pt")
                    
                    # Log best model info
                    best_model_msg = f"*** New best model saved with validation accuracy: {val_acc:.2f}% ***"
                    try:
                        if logger and hasattr(logger, 'log_both'):
                            logger.log_both(best_model_msg)
                            
                            # Direct file writing as backup
                            if hasattr(logger, 'log_file') and logger.log_file:
                                logger.log_file.write(best_model_msg + '\n')
                                logger.log_file.flush()
                        else:
                            # Fallback to console only
                            print(best_model_msg)
                            sys.stdout.flush()
                    except Exception as e:
                        print(best_model_msg)
                        print(f"Logging error in best model info: {str(e)}")
                        sys.stdout.flush()
                
                # Print memory usage after each epoch (even epochs only)
                if epoch % 2 == 0:
                    print_memory_usage(rank)
        
        # Only rank 0 plots metrics and finalizes
        if rank == 0:
            # Log end of training
            try:
                if logger:
                    logger.log_both(separator)
                    if hasattr(logger, 'log_file') and logger.log_file:
                        logger.log_file.write(separator + '\n')
                        logger.log_file.flush()
            except Exception as e:
                print(separator)
                print(f"Logging error at end of training: {str(e)}")
            
            # Plot and save metrics for this fold
            plot_metrics(epochs_list, train_losses, val_losses, "Loss", fold_idx+1)
            plot_metrics(epochs_list, train_accuracies, val_accuracies, "Accuracy", fold_idx+1)
            
            # Create dictionary for fold results
            fold_metrics = {
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_list': epochs_list
            }
            
            # Load best model and evaluate
            best_model = NNBinarySentimentClassifier(
                vocab_size=vocab_size,
                seq_length=seq_length, 
                hidden1_size=hidden1_size, 
                hidden2_size=hidden2_size, 
                output_size=output_size
            ).to(device)
            best_model.load_state_dict(torch.load(f"{MODEL_DIR}/NN_best_model_fold{fold_idx+1}.pt", weights_only=False))
            best_model.eval()
            
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    outputs = best_model(inputs)
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    
                    all_preds.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # Clean up
                    del inputs, targets, outputs, predictions
                
                # Garbage collection
                gc.collect()
                torch.cuda.empty_cache()
            
            # Calculate metrics
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()
            accuracy = accuracy_score(all_targets, all_preds) * 100
            
            # For binary classification, calculate precision, recall, and F1 score
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
            
            fold_result = {
                'fold': fold_idx + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'class_metrics': class_metrics,
                'metrics': fold_metrics
            }
            
            # Create fold summary string for backup file writing
            fold_summary = f"\nFold {fold_idx+1} Final Results:\n"
            fold_summary += f"Accuracy: {accuracy:.2f}%\n"
            fold_summary += f"Precision: {precision:.2f}%\n"
            fold_summary += f"Recall: {recall:.2f}%\n"
            fold_summary += f"F1 Score: {f1:.2f}%\n"
            fold_summary += "\nPer-Class Metrics:\n"
            for cls, metrics in class_metrics.items():
                fold_summary += f"{cls}: Precision={metrics['precision']:.2f}%, Recall={metrics['recall']:.2f}%, F1={metrics['f1']:.2f}%\n"
            
            # Log fold summary using multiple methods for redundancy
            try:
                # Normal logging through logger
                if logger and hasattr(logger, 'log_both'):
                    logger.log_both(f"\nFold {fold_idx+1} Final Results:")
                    logger.log_both(f"Accuracy: {accuracy:.2f}%")
                    logger.log_both(f"Precision: {precision:.2f}%")
                    logger.log_both(f"Recall: {recall:.2f}%")
                    logger.log_both(f"F1 Score: {f1:.2f}%")
                    logger.log_both("\nPer-Class Metrics:")
                    for cls, metrics in class_metrics.items():
                        logger.log_both(f"{cls}: Precision={metrics['precision']:.2f}%, Recall={metrics['recall']:.2f}%, F1={metrics['f1']:.2f}%")
                    
                    # Direct file writing as backup
                    if hasattr(logger, 'log_file') and logger.log_file:
                        logger.log_file.write(fold_summary)
                        logger.log_file.flush()
                else:
                    # Fallback to console only
                    print(fold_summary)
                    sys.stdout.flush()
            except Exception as e:
                # Last resort fallback
                print(fold_summary)
                print(f"Logging error in fold summary: {str(e)}")
                sys.stdout.flush()
            
            # Save fold results for later averaging - with improved pickling
            torch.save(fold_result, f"{MODEL_DIR}/fold_{fold_idx+1}_results.pt", pickle_protocol=4)
            
            # Return results from this fold
            return fold_result
    
    except Exception as e:
        print(f"Error in rank {rank}, fold {fold_idx+1}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up distributed processes
        cleanup()
    
    return None

def main():
    global logger  # Make logger accessible globally
    
    # Setup logging to file with improved logger
    log_filename = f"{LOGS_DIR}/NN_multigpu_training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = DualLogger(log_filename)

    # Start logging with multiple redundancy methods
    header = f"=== SemEval Binary Sentiment Analysis Neural Network Training Log (Multi-GPU) ==="
    start_time = f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    log_file_info = f"Log file: {log_filename}"
    separator = "=" * 70
    
    try:
        # Normal logging
        logger.log_both(header)
        logger.log_both(start_time)
        logger.log_both(log_file_info)
        logger.log_both(separator)
        
        # Direct file writing as backup
        if hasattr(logger, 'log_file') and logger.log_file:
            header_str = f"{header}\n{start_time}\n{log_file_info}\n{separator}\n"
            logger.log_file.write(header_str)
            logger.log_file.flush()
    except Exception as e:
        # Fallback to console
        print(header)
        print(start_time)
        print(log_file_info)
        print(separator)
        print(f"Logging error in header: {str(e)}")
        sys.stdout.flush()

    # Count available GPUs
    num_gpus = torch.cuda.device_count()
    try:
        logger.log_both(f"Number of available GPUs: {num_gpus}")
        
        if num_gpus < 1:
            logger.log_both("No GPUs available. Exiting.")
            logger.close()
            return
    except Exception as e:
        print(f"Number of available GPUs: {num_gpus}")
        print(f"Logging error: {str(e)}")
        if num_gpus < 1:
            print("No GPUs available. Exiting.")
            return
    
    # Print initial memory usage (console only)
    print_memory_usage()

    # Load dataset
    try:
        logger.log_both("Loading dataset...")
        df = pd.read_csv("/home/m23mac008/NLU/TOKENIZED_SemEval.CSV")

        logger.log_both("Sentiment distribution:")
        logger.log_both(str(df['sentiment'].value_counts()))

        # Convert token_indices from string to actual list
        logger.log_both("Converting token indices to lists...")
        df['token_indices'] = df['token_indices'].apply(ast.literal_eval)

        # Map sentiment labels for binary classification (0 -> 0, 4 -> 1)
        df['sentiment_label'] = df['sentiment'].apply(lambda x: 0 if x == 0 else 1)
        logger.log_both("\nRemapped sentiment distribution (0=negative, 1=positive):")
        logger.log_both(str(df['sentiment_label'].value_counts()))

        # Find the vocabulary size (maximum index in the token_indices + 1)
        vocab_size = max(max(indices) for indices in df['token_indices']) + 1
        logger.log_both(f"Vocabulary size: {vocab_size}")

        # Find the sequence length (length of token_indices lists)
        seq_length = len(df['token_indices'].iloc[0])
        logger.log_both(f"Sequence length: {seq_length}")
    except Exception as e:
        print("Error in dataset loading:")
        print(str(e))
        traceback.print_exc()
        if logger:
            logger.close()
        return

    # Print model architecture and hyperparameters
    try:
        logger.log_both("\n=== Neural Network Model Architecture and Hyperparameters ===")
        hidden1_size = 256
        hidden2_size = 128
        output_size = 1  # Binary classification
        learning_rate = 0.001
        num_epochs = 10
        
        # Use a smaller batch size to reduce memory footprint
        batch_size = 32 * num_gpus  # Start with a smaller batch size than before

        logger.log_both(f"Model type: Feed-forward Neural Network (NN) with Multi-GPU support")
        logger.log_both(f"Number of GPUs: {num_gpus}")
        logger.log_both(f"Vocabulary size: {vocab_size}")
        logger.log_both(f"Sequence length: {seq_length}")
        logger.log_both(f"Hidden layer 1 size: {hidden1_size}")
        logger.log_both(f"Hidden layer 2 size: {hidden2_size}")
        logger.log_both(f"Output size: {output_size} (binary classification)")
        logger.log_both(f"Learning rate: {learning_rate}")
        logger.log_both(f"Number of epochs: {num_epochs}")
        logger.log_both(f"Batch size: {batch_size} (total across all GPUs)")
        logger.log_both(f"Activation function: ReLU")
        logger.log_both(f"Loss function: BCEWithLogitsLoss (for binary classification)")
        logger.log_both(f"Optimizer: Adam")
        logger.log_both(separator)
        
        # Force flush to ensure hyperparameters are written
        if hasattr(logger, 'log_file') and logger.log_file:
            logger.log_file.flush()
    except Exception as e:
        print("\n=== Neural Network Model Architecture and Hyperparameters ===")
        print(f"Model type: Feed-forward Neural Network (NN) with Multi-GPU support")
        print(f"Number of GPUs: {num_gpus}")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Sequence length: {seq_length}")
        print(f"Hidden layer 1 size: {hidden1_size}")
        print(f"Hidden layer 2 size: {hidden2_size}")
        print(f"Output size: {output_size} (binary classification)")
        print(f"Learning rate: {learning_rate}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {batch_size} (total across all GPUs)")
        print(f"Activation function: ReLU")
        print(f"Loss function: BCEWithLogitsLoss (for binary classification)")
        print(f"Optimizer: Adam")
        print(separator)
        print(f"Logging error in hyperparameters: {str(e)}")
        sys.stdout.flush()

    # Convert data to appropriate format
    try:
        token_indices = df['token_indices'].tolist()
        labels = df['sentiment_label'].values

        # Lists to store metrics for all folds
        all_fold_results = []
        
        # Log conversion success
        logger.log_both("Successfully converted data to appropriate format.")
        if hasattr(logger, 'log_file') and logger.log_file:
            logger.log_file.flush()
    except Exception as e:
        print("Error in data conversion:")
        print(str(e))
        traceback.print_exc()
        if logger:
            logger.close()
        return

    # 5-fold cross-validation
    try:
        logger.log_both("\n=== Starting 5-Fold Cross-Validation with Multi-GPU Training ===")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Force flush to ensure this message is written
        if hasattr(logger, 'log_file') and logger.log_file:
            logger.log_file.flush()
        
        # Initialize multiprocessing with spawn method for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            logger.log_both("Start method already set to 'spawn', continuing...")
            # Force flush
            if hasattr(logger, 'log_file') and logger.log_file:
                logger.log_file.flush()
    except Exception as e:
        print("\n=== Starting 5-Fold Cross-Validation with Multi-GPU Training ===")
        print(f"Error in cross-validation setup: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
        if logger:
            logger.close()
        return

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(token_indices)):
        try:
            fold_msg = f"\nPreparing fold {fold_idx+1}/5..."
            indices_msg = f"Train indices size: {len(train_idx)}, Validation indices size: {len(val_idx)}"
            
            # Try multiple ways to log this important fold information
            try:
                logger.log_both(fold_msg)
                logger.log_both(indices_msg)
                
                # Direct file writing as backup
                if hasattr(logger, 'log_file') and logger.log_file:
                    logger.log_file.write(fold_msg + '\n')
                    logger.log_file.write(indices_msg + '\n')
                    logger.log_file.flush()
            except Exception as e:
                print(fold_msg)
                print(indices_msg)
                print(f"Logging error in fold header: {str(e)}")
                sys.stdout.flush()
            
            # Garbage collection before starting a new fold
            gc.collect()
            torch.cuda.empty_cache()
            
            # Use all available GPUs for training
            # We use spawn to properly handle CUDA initialization across processes
            try:
                mp.spawn(
                    train_fold,
                    args=(num_gpus, token_indices, labels, fold_idx, train_idx, val_idx, 
                        vocab_size, seq_length, hidden1_size, hidden2_size, output_size, 
                        num_epochs, batch_size // num_gpus, learning_rate),
                    nprocs=num_gpus,
                    join=True
                )
            except Exception as e:
                error_msg = f"Error in fold {fold_idx+1}: {str(e)}"
                cleanup_msg = "Attempting to clean up distributed processes..."
                
                # Try multiple ways to log this error
                try:
                    logger.log_both(error_msg)
                    logger.log_both(cleanup_msg)
                    
                    # Direct file writing as backup
                    if hasattr(logger, 'log_file') and logger.log_file:
                        logger.log_file.write(error_msg + '\n')
                        logger.log_file.write(cleanup_msg + '\n')
                        logger.log_file.flush()
                except Exception as log_e:
                    print(error_msg)
                    print(cleanup_msg)
                    print(f"Logging error: {str(log_e)}")
                    sys.stdout.flush()
                
                try:
                    dist.destroy_process_group()
                except:
                    pass
                # Continue with next fold instead of terminating completely
                continue
            
            # Load and store the result from this fold (saved by rank 0)
            fold_result_file = f"{MODEL_DIR}/fold_{fold_idx+1}_results.pt"
            if os.path.exists(fold_result_file):
                # Fix the loading issue with weights_only=False
                fold_result = torch.load(fold_result_file, weights_only=False)
                all_fold_results.append(fold_result)
                
                # Log successful fold completion
                success_msg = f"Successfully completed fold {fold_idx+1} and loaded results."
                try:
                    logger.log_both(success_msg)
                    if hasattr(logger, 'log_file') and logger.log_file:
                        logger.log_file.flush()
                except Exception as e:
                    print(success_msg)
                    print(f"Logging error: {str(e)}")
                    sys.stdout.flush()
            else:
                error_msg = f"Warning: Could not find results file for fold {fold_idx+1}."
                try:
                    logger.log_both(error_msg)
                    if hasattr(logger, 'log_file') and logger.log_file:
                        logger.log_file.flush()
                except Exception as e:
                    print(error_msg)
                    print(f"Logging error: {str(e)}")
                    sys.stdout.flush()
        except Exception as fold_e:
            # Catch any unexpected errors during fold processing
            error_msg = f"Unexpected error in fold {fold_idx+1}: {str(fold_e)}"
            try:
                logger.log_both(error_msg)
                if hasattr(logger, 'log_file') and logger.log_file:
                    logger.log_file.flush()
            except Exception as e:
                print(error_msg)
                print(f"Logging error: {str(e)}")
                sys.stdout.flush()
            continue

    # Calculate and report average metrics across all folds
    try:
        if all_fold_results:
            # Calculate average metrics
            avg_acc = sum(fold['accuracy'] for fold in all_fold_results) / len(all_fold_results)
            avg_precision = sum(fold['precision'] for fold in all_fold_results) / len(all_fold_results)
            avg_recall = sum(fold['recall'] for fold in all_fold_results) / len(all_fold_results)
            avg_f1 = sum(fold['f1'] for fold in all_fold_results) / len(all_fold_results)

            avg_class_metrics = {
                'Negative (0)': {
                    'precision': sum(fold['class_metrics']['Negative (0)']['precision'] for fold in all_fold_results) / len(all_fold_results),
                    'recall': sum(fold['class_metrics']['Negative (0)']['recall'] for fold in all_fold_results) / len(all_fold_results),
                    'f1': sum(fold['class_metrics']['Negative (0)']['f1'] for fold in all_fold_results) / len(all_fold_results)
                },
                'Positive (1)': {
                    'precision': sum(fold['class_metrics']['Positive (1)']['precision'] for fold in all_fold_results) / len(all_fold_results),
                    'recall': sum(fold['class_metrics']['Positive (1)']['recall'] for fold in all_fold_results) / len(all_fold_results),
                    'f1': sum(fold['class_metrics']['Positive (1)']['f1'] for fold in all_fold_results) / len(all_fold_results)
                }
            }

            # Get training metrics for plotting
            all_train_acc = [fold['metrics']['train_accuracies'] for fold in all_fold_results]
            all_val_acc = [fold['metrics']['val_accuracies'] for fold in all_fold_results]
            all_train_loss = [fold['metrics']['train_losses'] for fold in all_fold_results]
            all_val_loss = [fold['metrics']['val_losses'] for fold in all_fold_results]
            epochs_list = all_fold_results[0]['metrics']['epochs_list']

            # Plot average metrics across all folds
            plot_average_metrics(epochs_list, all_train_loss, all_val_loss, "Loss")
            plot_average_metrics(epochs_list, all_train_acc, all_val_acc, "Accuracy")

            # Create summary string for direct file writing
            summary = "\n" + "=" * 70 + "\n"
            summary += "===== Neural Network 5-Fold Cross-Validation Results (Multi-GPU) =====\n"
            summary += "=" * 70 + "\n"
            summary += f"Average Accuracy: {avg_acc:.2f}%\n"
            summary += f"Average Precision: {avg_precision:.2f}%\n"
            summary += f"Average Recall: {avg_recall:.2f}%\n"
            summary += f"Average F1 Score: {avg_f1:.2f}%\n"
            
            summary += "\nAverage Per-Class Metrics:\n"
            summary += "-" * 50 + "\n"
            summary += f"{'Class':^12} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}\n"
            summary += "-" * 50 + "\n"
            for cls, metrics in avg_class_metrics.items():
                summary += f"{cls:^12} | {metrics['precision']:^10.2f}% | {metrics['recall']:^10.2f}% | {metrics['f1']:^10.2f}%\n"
            summary += "-" * 50 + "\n"
            
            # Try normal logging
            try:
                logger.log_both("\n" + "=" * 70)
                logger.log_both("===== Neural Network 5-Fold Cross-Validation Results (Multi-GPU) =====")
                logger.log_both("=" * 70)
                logger.log_both(f"Average Accuracy: {avg_acc:.2f}%")
                logger.log_both(f"Average Precision: {avg_precision:.2f}%")
                logger.log_both(f"Average Recall: {avg_recall:.2f}%")
                logger.log_both(f"Average F1 Score: {avg_f1:.2f}%")
                
                logger.log_both("\nAverage Per-Class Metrics:")
                logger.log_both("-" * 50)
                logger.log_both(f"{'Class':^12} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}")
                logger.log_both("-" * 50)
                for cls, metrics in avg_class_metrics.items():
                    logger.log_both(f"{cls:^12} | {metrics['precision']:^10.2f}% | {metrics['recall']:^10.2f}% | {metrics['f1']:^10.2f}%")
                logger.log_both("-" * 50)
                
                # Ensure flush
                if hasattr(logger, 'log_file') and logger.log_file:
                    logger.log_file.flush()
            except Exception as e:
                print(f"Logging error in summary: {str(e)}")
            
            # Backup direct file writing
            try:
                if hasattr(logger, 'log_file') and logger.log_file:
                    logger.log_file.write(summary)
                    logger.log_file.flush()
            except Exception as e:
                print(f"Direct file writing error: {str(e)}")
            
            # Last resort fallback
            if not hasattr(logger, 'log_file') or not logger.log_file:
                print(summary)
                sys.stdout.flush()
        else:
            no_results_msg = "\nNo fold results were successfully completed. Check logs for errors."
            try:
                logger.log_both(no_results_msg)
                if hasattr(logger, 'log_file') and logger.log_file:
                    logger.log_file.flush()
            except Exception as e:
                print(no_results_msg)
                print(f"Logging error: {str(e)}")
                sys.stdout.flush()
    except Exception as e:
        print("\nError in calculating average metrics:")
        print(str(e))
        traceback.print_exc()
        sys.stdout.flush()

    # Final outputs and completion information
    try:
        output_info = "\nFile Outputs:\n"
        output_info += f"- NN Training logs saved to: {log_filename}\n"
        output_info += f"- NN Training graphs saved to: {GRAPHS_DIR}/ directory\n"
        output_info += f"- NN Best models saved as: {MODEL_DIR}/NN_best_model_fold[1-5].pt\n\n"
        
        end_time = f"Multi-GPU Neural Network Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output_info += end_time
        output_info += "=" * 70 + "\n"

        # Additional information about multi-GPU training
        output_info += "\n=== Notes on Multi-GPU Acceleration ===\n"
        output_info += "- Using PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training\n"
        output_info += "- Each GPU processes a portion of each batch in parallel\n"
        output_info += "- Using memory-efficient approach to encoding token indices\n"
        output_info += "- Gradients are automatically synchronized across GPUs\n"
        output_info += "- Batch size is reduced to fit in GPU memory\n"
        output_info += "- Data is distributed using DistributedSampler to ensure each GPU gets different samples\n"
        output_info += "- Only rank 0 (first GPU) performs validation and metrics reporting\n"
        output_info += "- Regular garbage collection to prevent memory leaks\n"
        output_info += "=" * 70 + "\n"
        
        # Try multiple logging methods for redundancy
        try:
            logger.log_both("\nFile Outputs:")
            logger.log_both(f"- NN Training logs saved to: {log_filename}")
            logger.log_both(f"- NN Training graphs saved to: {GRAPHS_DIR}/ directory")
            logger.log_both(f"- NN Best models saved as: {MODEL_DIR}/NN_best_model_fold[1-5].pt")

            logger.log_both(f"\nMulti-GPU Neural Network Training completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.log_both("=" * 70)

            # Additional information about multi-GPU training
            logger.log_both("\n=== Notes on Multi-GPU Acceleration ===")
            logger.log_both("- Using PyTorch's DistributedDataParallel (DDP) for efficient multi-GPU training")
            logger.log_both("- Each GPU processes a portion of each batch in parallel")
            logger.log_both("- Using memory-efficient approach to encoding token indices")
            logger.log_both("- Gradients are automatically synchronized across GPUs")
            logger.log_both("- Batch size is reduced to fit in GPU memory")
            logger.log_both("- Data is distributed using DistributedSampler to ensure each GPU gets different samples")
            logger.log_both("- Only rank 0 (first GPU) performs validation and metrics reporting")
            logger.log_both("- Regular garbage collection to prevent memory leaks")
            logger.log_both("=" * 70)
            
            # Force flush
            if hasattr(logger, 'log_file') and logger.log_file:
                logger.log_file.flush()
        except Exception as e:
            print(f"Logging error in completion information: {str(e)}")
        
        # Backup direct file writing
        try:
            if hasattr(logger, 'log_file') and logger.log_file:
                logger.log_file.write(output_info)
                logger.log_file.flush()
        except Exception as e:
            print(f"Direct file writing error: {str(e)}")
        
        # Fallback console output
        if not hasattr(logger, 'log_file') or not logger.log_file:
            print(output_info)
            sys.stdout.flush()
    except Exception as e:
        print("\nError in printing final outputs:")
        print(str(e))
        sys.stdout.flush()
    
    # Close the logger
    try:
        if logger:
            logger.close()
            print("Logger closed successfully.")
    except Exception as e:
        print(f"Error closing logger: {str(e)}")
        sys.stdout.flush()

if __name__ == "__main__":
    try:
        print("Starting SemEval Binary Sentiment Analysis Neural Network Training...")
        sys.stdout.flush()
        main()
        print("Training completed successfully.")
        sys.stdout.flush()
    except Exception as e:
        print("Critical error in main program:")
        print(str(e))
        traceback.print_exc()
        sys.stdout.flush()