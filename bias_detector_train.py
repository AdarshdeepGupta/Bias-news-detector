import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING & PREPROCESSING ====================

def load_and_preprocess_data():
    """Load dataset and perform initial preprocessing"""
    print("Loading dataset...")
    df = pd.read_csv("hf://datasets/cajcodes/political-bias/political_bias.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Drop rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])
    
    print(f"\nAfter cleaning: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Simple tokenization"""
    return text.split()

# ==================== WORD2VEC EMBEDDINGS ====================

def train_word2vec(sentences, vector_size=100, window=5, min_count=2, workers=4):
    """Train Word2Vec model on the corpus"""
    print("\nTraining Word2Vec model...")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram model
        epochs=10
    )
    
    print(f"Vocabulary size: {len(model.wv)}")
    
    return model

def create_embedding_matrix(word2vec_model, vocab, embedding_dim):
    """Create embedding matrix from Word2Vec model"""
    vocab_size = len(vocab) + 1  # +1 for padding token
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in vocab.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]
        else:
            # Random initialization for unknown words
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
    
    return embedding_matrix

# ==================== SEQUENCE PROCESSING ====================

def build_vocabulary(tokenized_texts):
    """Build vocabulary from tokenized texts"""
    vocab = {}
    idx = 1  # 0 reserved for padding
    
    for tokens in tokenized_texts:
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    
    return vocab

def texts_to_sequences(tokenized_texts, vocab):
    """Convert tokenized texts to sequences of integers"""
    sequences = []
    for tokens in tokenized_texts:
        seq = [vocab.get(token, 0) for token in tokens]
        sequences.append(seq)
    
    return sequences

# ==================== CNN MODEL ====================

def build_cnn_model(vocab_size, embedding_dim, embedding_matrix, max_length, num_classes):
    """Build CNN model for text classification"""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=True  # Fine-tune embeddings
        ),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==================== MAIN TRAINING PIPELINE ====================

def main():
    # Configuration
    MAX_LENGTH = 200
    EMBEDDING_DIM = 100
    TEST_SIZE = 0.2
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # 1. Load and preprocess data
    df = load_and_preprocess_data()
    
    # 2. Clean text
    print("\nCleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 3. Tokenize
    print("Tokenizing...")
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    
    # 4. Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    print(f"Label mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {idx}")
    
    num_classes = len(label_encoder.classes_)
    
    # 5. Train Word2Vec
    tokenized_texts = df['tokens'].tolist()
    word2vec_model = train_word2vec(tokenized_texts, vector_size=EMBEDDING_DIM)
    
    # 6. Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocabulary(tokenized_texts)
    vocab_size = len(vocab) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # 7. Create embedding matrix
    print("\nCreating embedding matrix...")
    embedding_matrix = create_embedding_matrix(word2vec_model, vocab, EMBEDDING_DIM)
    
    # 8. Convert texts to sequences
    print("\nConverting texts to sequences...")
    sequences = texts_to_sequences(tokenized_texts, vocab)
    
    # 9. Pad sequences
    print("Padding sequences...")
    X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    y = df['label_encoded'].values
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # 10. Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 11. Build and train CNN model
    print("\nBuilding CNN model...")
    model = build_cnn_model(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_LENGTH, num_classes)
    
    print("\nModel architecture:")
    model.summary()
    
    print("\nTraining model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 12. Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 13. Save everything
    print("\nSaving model and artifacts...")
    
    # Save CNN model
    model.save('bias_detector_model.h5')
    
    # Save vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save Word2Vec model
    word2vec_model.save('word2vec_model.bin')
    
    # Save config
    config = {
        'max_length': MAX_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'vocab_size': vocab_size,
        'num_classes': num_classes
    }
    with open('config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print("\n✓ Training complete! All files saved.")
    print("Files created:")
    print("  - bias_detector_model.h5")
    print("  - vocab.pkl")
    print("  - label_encoder.pkl")
    print("  - word2vec_model.bin")
    print("  - config.pkl")

if __name__ == "__main__":
    main()
