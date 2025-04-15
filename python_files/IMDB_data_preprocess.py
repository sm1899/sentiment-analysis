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

# Enable tqdm for pandas
tqdm.pandas()

# Set up paths - MODIFY THESE TO MATCH YOUR SETUP
IMDB_ROOT_DIR = "/home/m23mac008/NLU/IMDB/aclImdb"  

# Specific directories based on IMDB structure
TRAIN_DIR = os.path.join(IMDB_ROOT_DIR, "train")
TEST_DIR = os.path.join(IMDB_ROOT_DIR, "test")

def load_reviews_from_folder(folder_path, sentiment_value):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, encoding='utf-8') as f:
            text = f.read().strip()
            data.append({'sentiment': sentiment_value, 'text': text})
    return data

train_pos_data = load_reviews_from_folder(os.path.join(TRAIN_DIR, 'pos'), 1)
train_neg_data = load_reviews_from_folder(os.path.join(TRAIN_DIR, 'neg'), 0)
train_data = train_pos_data + train_neg_data
train_df = pd.DataFrame(train_data)
train_df  = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

test_pos_data = load_reviews_from_folder(os.path.join(TEST_DIR, 'pos'), 1)
test_neg_data = load_reviews_from_folder(os.path.join(TEST_DIR, 'neg'), 0)
test_data = test_pos_data + test_neg_data
test_df = pd.DataFrame(test_data)
test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)



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

    def setup_workers_and_chunks(df, dataset_name):
        chunk_size = len(df) // num_workers
        return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        print("No GPUs, using CPU")

    num_workers = num_gpus if num_gpus > 0 else min(os.cpu_count(), 8)
    print(f"Using {num_workers} {'GPU' if num_gpus > 0 else 'CPU'} workers")

    # Phase 1: Tokenize & build vocab on training data
    print("\n📥 Processing TRAIN set...")
    train_chunks = setup_workers_and_chunks(train_df, "train")

    word_counter = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(train_chunks):
            gpu_id = i % num_gpus if num_gpus > 0 else None
            futures.append(executor.submit(process_chunk, chunk, i, gpu_id=gpu_id))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Counting words"):
            chunk_counter = future.result()
            word_counter.update(chunk_counter)

    freq_threshold = 5
    vocab = {word for word, count in word_counter.items() if count >= freq_threshold}
    vocab.update(['UNK', 'PAD'])

    word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    total_words = sum(word_counter.values())
    max_length = int(total_words / len(train_df))

    vocab_info = {
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'max_length': max_length,
        'vocab_size': len(vocab)
    }

    with open('IMBD_vocab.pkl', 'wb') as f:
        pickle.dump(vocab_info, f)

    # Phase 2: Tokenize & index training set
    print("\n📦 Tokenizing & indexing TRAIN set...")
    train_processed_chunks = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(train_chunks):
            gpu_id = i % num_gpus if num_gpus > 0 else None
            futures.append(executor.submit(
                process_chunk, chunk, i, vocab, max_length, word_to_idx, gpu_id
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing train"):
            train_processed_chunks.append(future.result())

    tokenized_train_df = pd.concat(train_processed_chunks, ignore_index=True)
    tokenized_train_df.to_csv("TOKENIZED_Train_IMBD.CSV", index=False)
    tokenized_train_df.to_pickle("TOKENIZED_Train_IMBD.pkl")

    # Phase 3: Tokenize & index testing set
    print("\n📦 Tokenizing & indexing TEST set...")
    test_chunks = setup_workers_and_chunks(test_df, "test")
    test_processed_chunks = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, chunk in enumerate(test_chunks):
            gpu_id = i % num_gpus if num_gpus > 0 else None
            futures.append(executor.submit(
                process_chunk, chunk, i, vocab, max_length, word_to_idx, gpu_id
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing test"):
            test_processed_chunks.append(future.result())

    tokenized_test_df = pd.concat(test_processed_chunks, ignore_index=True)
    tokenized_test_df.to_csv("TOKENIZED_Test_IMBD.CSV", index=False)
    tokenized_test_df.to_pickle("TOKENIZED_Test_IMBD.pkl")

    elapsed = time.time() - start_time
    print(f"\n✅ Finished processing IMDB dataset in {elapsed // 60:.0f}m {elapsed % 60:.2f}s")
    print("Saved files:")
    print("- TOKENIZED_Train_IMBD.CSV")
    print("- TOKENIZED_Train_IMBD.pkl")
    print("- TOKENIZED_Test_IMBD.CSV")
    print("- TOKENIZED_Test_IMBD.pkl")
    print("- IMBD_vocab.pkl")



if __name__ == "__main__":
    # Initialize multiprocessing with spawn method for better compatibility with CUDA
    mp.set_start_method('spawn', force=True)
    main()