import pandas as pd
import spacy
from tqdm import tqdm
from collections import Counter
import numpy as np
import pickle

# tqdm settings for pandas
tqdm.pandas()

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# 📥 Load dataset
print("📥 Loading dataset...")
df = pd.read_csv('/home/m23mac008/NLU/twitter_training.csv', header=None)

# Select only columns 2 (sentiment) and 3 (text)
df = df[[2, 3]]

# Rename columns for clarity
df.columns = ['sentiment', 'text']

# Remove 'Irrelevant' sentiment rows
df = df[df['sentiment'] != 'Irrelevant']

# Drop rows where 'text' is empty or NaN
df = df.dropna(subset=['text'])

# Convert 'text' to string to avoid errors
df['text'] = df['text'].astype(str)

# Reset index after cleaning
df = df.reset_index(drop=True)

print(f"Dataset loaded. Shape: {df.shape}")
print(df.head())

# Step 1: Use spaCy English tokenizer
print("\n1. Tokenizing with spaCy...")

def tokenize_text(text):
    """Tokenize text using spaCy, removing punctuation"""
    doc = nlp(text)
    # Get tokens, excluding punctuation
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return tokens

# Apply tokenization to the entire dataset
df['tokens'] = df['text'].progress_apply(tokenize_text)

# Count word frequencies
all_tokens = []
for tokens in df['tokens']:
    all_tokens.extend(tokens)

word_counts = Counter(all_tokens)
print(f"Total unique words before filtering: {len(word_counts)}")

# Step 2: Filter words by frequency and replace with UNK
print("\n2. Filtering words by frequency and replacing infrequent words with UNK...")
freq_threshold = 5
vocab = {word for word, count in word_counts.items() if count >= freq_threshold}

# Add special tokens
vocab.add('UNK')  # Unknown token
vocab.add('PAD')  # Padding token

print(f"Words with frequency >= {freq_threshold}: {len(vocab) - 2}")  # -2 for UNK and PAD

# Function to replace infrequent words with UNK
def replace_infrequent_words(tokens):
    return [token if token in vocab else 'UNK' for token in tokens]

df['filtered_tokens'] = df['tokens'].progress_apply(replace_infrequent_words)

# Step 3: Calculate statistics for padding/truncating
total_words = sum(len(tokens) for tokens in df['tokens'])
num_sentences = len(df)
avg_length = total_words / num_sentences
max_length = int(avg_length)

print(f"\nStatistics:")
print(f"Total number of words (tokens): {total_words}")
print(f"Number of sentences: {num_sentences}")
print(f"Average length: {avg_length:.2f}")
print(f"Setting maximum length to: {max_length}")
print(f"Vocabulary size: {len(vocab)}")

# Step 4: Pad or truncate sentences
print("\n3. Padding/truncating sentences to fixed length...")
def pad_or_truncate(tokens, max_len):
    """Truncate if longer than max_len, pad with 'PAD' if shorter"""
    if len(tokens) > max_len:
        return tokens[:max_len]
    else:
        return tokens + ['PAD'] * (max_len - len(tokens))

df['padded_tokens'] = df['filtered_tokens'].progress_apply(lambda x: pad_or_truncate(x, max_length))

# Step 5: Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}

print("\nWord-to-index mapping (first 10 entries):")
print(list(word_to_idx.items())[:10])

# Step 6: Convert tokens to indices
print("\n4. Converting tokens to their numerical indices...")
def tokens_to_indices(tokens):
    return [word_to_idx.get(token, word_to_idx['UNK']) for token in tokens]

df['token_indices'] = df['padded_tokens'].progress_apply(tokens_to_indices)

# Create final DataFrame with sentiment, original text, and token indices
tokenized_df = df[['sentiment', 'text', 'token_indices']]

# Save to CSV 
print("\nSaving tokenized data to CSV...")
tokenized_df.to_csv('TOKENIZED_TWITTER.CSV', index=False)

# Also save as pickle to preserve list structure of indices
tokenized_df.to_pickle('TOKENIZED_TWITTER.pkl')

# Save vocabulary information
vocab_info = {
    'vocab': vocab,
    'word_to_idx': word_to_idx,
    'idx_to_word': {idx: word for word, idx in word_to_idx.items()},
    'max_length': max_length,
    'vocab_size': len(vocab)
}

with open('twitter_vocab.pkl', 'wb') as f:
    pickle.dump(vocab_info, f)

print("\nFinal processed data (first 3 rows):")
print(tokenized_df.head(3))

# Print statistics
print(f"\nVocabulary size: {len(vocab)}")
print(f"Maximum sequence length: {max_length}")

print("\nProcessing complete! ✅")
print("Files saved:")
print("- TOKENIZED_TWITTER.CSV (with token indices)")
print("- TOKENIZED_TWITTER.pkl (with token indices as proper lists)")
print("- twitter_vocab.pkl (vocabulary information)")


















# # Load spaCy English model
# nlp = spacy.load("en_core_web_sm")

# # Function to tokenize text
# def tokenize_text(text):
#     doc = nlp(text)
#     return [token.text.lower() for token in doc] 

# # Apply tokenization with tqdm progress bar
# df["tokenized_text"] = df["text"].progress_apply(tokenize_text)

# # Compute word frequencies
# word_counts = Counter([word for tokens in df["tokenized_text"] for word in tokens])

# # Replace rare words (frequency < 5) with "UNK"
# def replace_rare_words(tokens, min_freq=5):
#     return [word if word_counts[word] >= min_freq else "UNK" for word in tokens]

# df["tokenized_text"] = df["tokenized_text"].progress_apply(replace_rare_words)

# # Compute average sentence length (for padding/truncation)
# avg_length = int(sum(len(tokens) for tokens in df["tokenized_text"]) / len(df["tokenized_text"]))
# print(f"📏 Setting max sequence length to: {avg_length}")

# # Function to pad/truncate sentences
# def pad_or_truncate(tokens, max_length=avg_length):
#     if len(tokens) > max_length:
#         return tokens[:max_length]  # Truncate
#     else:
#         return tokens + ["PAD"] * (max_length - len(tokens))  # Pad with "PAD"

# df["padded_text"] = df["tokenized_text"].progress_apply(lambda tokens: pad_or_truncate(tokens, avg_length))

# # 🔹 Print first few processed samples
# print(df.head())

# # 🔹 Save preprocessed dataset
# df.to_csv("preprocessed_twitter_data.csv", index=False)


# # ✅ Print the first few rows to verify
# print(df.head())

# # Optional: Save processed DataFrame
# df.to_csv("/home/m23mac008/NLU/tokenized_twitter_data.csv", index=False)
