import pandas as pd
import spacy
from tqdm import tqdm
from collections import Counter
import numpy as np
import pickle
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from functools import partial
import warnings
import time

# Disable warnings
warnings.filterwarnings('ignore')

# Enable tqdm for pandas
tqdm.pandas()

def initialize_gpu(gpu_id):
    """Initialize a GPU device and set up spaCy to use it"""
    try:
        if torch.cuda.is_available():
            # Set the CUDA device
            torch.cuda.set_device(gpu_id)
            print(f"Process using GPU: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
            
            # Enable GPU for spaCy
            spacy.require_gpu(gpu_id)
            print(f"spaCy using GPU: {gpu_id}")
        else:
            print("No GPU available, using CPU")
    except Exception as e:
        print(f"Error initializing GPU: {e}")
        print("Falling back to CPU")
    
    # Load the spaCy model with optimized settings for batch processing
    nlp = spacy.load("en_core_web_sm")
    
    # Disable components we don't need for tokenization to speed up processing
    nlp.select_pipes(enable=["tok2vec", "tagger"])
    
    return nlp

def batch_tokenize(texts, nlp, batch_size=1000):
    """Tokenize texts in batches using spaCy's pipe functionality"""
    all_tokens = []
    
    # Progress bar for the batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Process in batches with a progress bar
    with tqdm(total=len(texts), desc="Tokenizing batch") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process the batch using spaCy's pipe for efficiency
            docs = list(nlp.pipe(batch_texts, batch_size=batch_size))
            
            # Extract tokens for each document
            batch_tokens = []
            for doc in docs:
                # Get tokens, excluding punctuation
                tokens = [token.text.lower() for token in doc if not token.is_punct]
                batch_tokens.append(tokens)
                
            all_tokens.extend(batch_tokens)
            pbar.update(len(batch_texts))
    
    return all_tokens

def process_chunk(chunk_df, chunk_id, vocab=None, max_length=None, word_to_idx=None, gpu_id=None):
    """Process a chunk of the dataframe with efficient batch processing"""
    start_time = time.time()
    print(f"Worker {chunk_id} starting on {'GPU:'+str(gpu_id) if gpu_id is not None else 'CPU'}")
    
    # Initialize GPU and load spaCy model
    nlp = initialize_gpu(gpu_id) if gpu_id is not None else spacy.load("en_core_web_sm")
    
    # Get texts as a list
    texts = chunk_df['text'].tolist()
    
    # Determine optimal batch size based on available GPU memory
    # RTX A6000 and A5000 have plenty of memory, so we can use larger batches
    batch_size = 2000 if gpu_id is not None else 1000
    
    # Tokenize the texts in batches
    tokens_list = batch_tokenize(texts, nlp, batch_size)
    
    # Assign tokens to dataframe
    chunk_df['tokens'] = tokens_list
    
    # If vocab is not provided, we're in the first phase (counting words)
    if vocab is None:
        # Count word frequencies in this chunk
        all_tokens = []
        for tokens in tokens_list:
            all_tokens.extend(tokens)
        word_counter = Counter(all_tokens)
        elapsed = time.time() - start_time
        print(f"Worker {chunk_id} completed word counting in {elapsed:.2f} seconds")
        return word_counter
    
    # Otherwise, we're in the second phase with a known vocabulary
    # Use vectorized operations for better performance
    
    # Replace infrequent words with UNK
    print(f"Worker {chunk_id}: Filtering tokens")
    chunk_df['filtered_tokens'] = [
        [token if token in vocab else 'UNK' for token in tokens]
        for tokens in chunk_df['tokens']
    ]
    
    # Pad or truncate sentences
    print(f"Worker {chunk_id}: Padding/truncating tokens")
    chunk_df['padded_tokens'] = [
        tokens[:max_length] if len(tokens) > max_length else tokens + ['PAD'] * (max_length - len(tokens))
        for tokens in chunk_df['filtered_tokens']
    ]
    
    # Convert tokens to indices
    print(f"Worker {chunk_id}: Converting to indices")
    chunk_df['token_indices'] = [
        [word_to_idx.get(token, word_to_idx['UNK']) for token in tokens]
        for tokens in chunk_df['padded_tokens']
    ]
    
    # Get only the columns we need for the final result
    result = chunk_df[['sentiment', 'text', 'token_indices']]
    
    elapsed = time.time() - start_time
    print(f"Worker {chunk_id} completed processing in {elapsed:.2f} seconds")
    return result

def main():
    start_time = time.time()
    
    # Check for GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        print("No GPUs available, using CPU only")
    
    # Set the number of workers based on available GPUs and CPU cores
    if num_gpus > 0:
        num_workers = num_gpus
        print(f"Using {num_workers} GPU workers")
    else:
        num_workers = min(os.cpu_count(), 8)  # Limit to 8 CPU workers to avoid excessive overhead
        print(f"Using {num_workers} CPU workers")
    
    # Load dataset
    print("\n📥 Loading dataset...")
    df = pd.read_csv('/home/m23mac008/NLU/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['sentiment', 'text']
    
    # Clean dataset
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    df = df.reset_index(drop=True)
    
    print(f"Dataset loaded. Shape: {df.shape}")
    print(df.head())
    
    # Split dataframe into chunks for parallel processing
    # Adjust chunk_size to ensure even distribution across workers
    chunk_size = len(df) // num_workers
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    print(f"\nSplitting data into {len(chunks)} chunks for parallel processing")
    
    # Phase 1: Count word frequencies across all chunks
    print("\n1. Tokenizing with spaCy and counting word frequencies...")
    word_counter = Counter()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Assign each chunk to a GPU (or CPU if not enough GPUs)
        futures = []
        for i, chunk in enumerate(chunks):
            # Assign GPU ID if available, otherwise None (CPU)
            gpu_id = i % num_gpus if num_gpus > 0 else None
            futures.append(executor.submit(process_chunk, chunk, i, gpu_id=gpu_id))
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            chunk_counter = future.result()
            word_counter.update(chunk_counter)
    
    print(f"Total unique words before filtering: {len(word_counter)}")
    
    # Step 2: Filter words by frequency
    print("\n2. Filtering words by frequency...")
    freq_threshold = 5
    vocab = {word for word, count in word_counter.items() if count >= freq_threshold}
    
    # Add special tokens
    vocab.add('UNK')  # Unknown token
    vocab.add('PAD')  # Padding token
    
    print(f"Words with frequency >= {freq_threshold}: {len(vocab) - 2}")  # -2 for UNK and PAD
    
    # Step 3: Calculate statistics for padding/truncating
    total_words = sum(count for word, count in word_counter.items())
    num_sentences = len(df)
    avg_length = total_words / num_sentences
    max_length = int(avg_length)
    
    print(f"\nStatistics:")
    print(f"Total number of words (tokens): {total_words}")
    print(f"Number of sentences: {num_sentences}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Setting maximum length to: {max_length}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Step 4: Create word-to-index mapping
    word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    
    print("\nWord-to-index mapping (first 10 entries):")
    print(list(word_to_idx.items())[:10])
    
    # Phase 2: Process each chunk with the known vocabulary
    print("\n3. Processing chunks with the vocabulary...")
    processed_chunks = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Assign each chunk to a GPU (or CPU if not enough GPUs)
        futures = []
        for i, chunk in enumerate(chunks):
            # Assign GPU ID if available, otherwise None (CPU)
            gpu_id = i % num_gpus if num_gpus > 0 else None
            futures.append(executor.submit(
                process_chunk, 
                chunk, 
                i, 
                vocab=vocab, 
                max_length=max_length, 
                word_to_idx=word_to_idx, 
                gpu_id=gpu_id
            ))
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            chunk_result = future.result()
            processed_chunks.append(chunk_result)
    
    # Combine processed chunks into a single dataframe
    print("\n4. Combining results...")
    tokenized_df = pd.concat(processed_chunks, ignore_index=True)
    
    # Save to CSV 
    print("\nSaving tokenized data to CSV...")
    tokenized_df.to_csv('TOKENIZED_SemEval.CSV', index=False)
    
    # Also save as pickle to preserve list structure of indices
    tokenized_df.to_pickle('TOKENIZED_SemEval.pkl')
    
    # Save vocabulary information
    vocab_info = {
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'idx_to_word': {idx: word for word, idx in word_to_idx.items()},
        'max_length': max_length,
        'vocab_size': len(vocab)
    }
    
    with open('SemEval_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_info, f)
    
    print("\nFinal processed data (first 3 rows):")
    print(tokenized_df.head(3))
    
    # Print statistics
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Maximum sequence length: {max_length}")
    
    # Calculate and print total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    print("\nProcessing complete! ✅")
    print("Files saved:")
    print("- TOKENIZED_SemEval.CSV (with token indices)")
    print("- TOKENIZED_SemEval.pkl (with token indices as proper lists)")
    print("- SemEval_vocab.pkl (vocabulary information)")

if __name__ == "__main__":
    # Initialize multiprocessing with spawn method for better compatibility with CUDA
    mp.set_start_method('spawn', force=True)
    main()
