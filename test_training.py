"""Quick test script to diagnose training issues"""
import time
import numpy as np
import tensorflow as tf
from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data, feature_engineering_tf_text_vectorization
from src.models import train_model

print("=" * 60)
print("DIAGNOSTIC TEST SCRIPT")
print("=" * 60)

# Check TensorFlow setup
print("\n1. Checking TensorFlow setup...")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"CPU Available: {tf.config.list_physical_devices('CPU')}")

# Load small subset of data
print("\n2. Loading data...")
start = time.time()
train, test = load_data()
print(f"   Loaded in {time.time() - start:.2f}s")
print(f"   Train shape: {train.shape}")

# Use only 1000 samples for quick test
print("\n3. Creating small subset (1000 samples)...")
train_small = train.sample(n=1000, random_state=42).reset_index(drop=True)
train_small, dev_small = split_dataset(train_small)
print(f"   Train subset: {train_small.shape}")
print(f"   Dev subset: {dev_small.shape}")

# Preprocess
print("\n4. Preprocessing...")
start = time.time()
train_small = preprocess_data(train_small)
dev_small = preprocess_data(dev_small)
print(f"   Preprocessed in {time.time() - start:.2f}s")

# Feature engineering
print("\n5. Feature engineering...")
start = time.time()
X_train, vectorizer = feature_engineering_tf_text_vectorization(
    train_small, 
    column_name="description", 
    max_tokens=5000,
    output_sequence_length=128
)
y_train = train_small['label'].values - 1  # Convert from 1-indexed to 0-indexed
print(f"   Vectorized in {time.time() - start:.2f}s")
print(f"   X_train type: {type(X_train)}, shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"   y_train type: {type(y_train)}, shape: {y_train.shape}, dtype: {y_train.dtype}")

X_dev = feature_engineering_tf_text_vectorization(
    dev_small, 
    column_name="description",
    max_tokens=5000,
    output_sequence_length=128,
    vectorizer=vectorizer
)
y_dev = dev_small['label'].values - 1  # Convert from 1-indexed to 0-indexed

# Check label values
print(f"\n6. Checking label values...")
print(f"   Unique labels in y_train: {np.unique(y_train)}")
print(f"   Label counts: {np.bincount(y_train)}")

# Test CNN model creation
print("\n7. Building CNN model...")
start = time.time()
cnn_model = train_model(
    'cnn',
    X_train,
    y_train,
    X_val=X_dev,
    y_val=y_dev,
    vocab_size=5000,
    embed_dim=64,  # Smaller for faster test
    epochs=1,
    batch_size=32
)
print(f"   Training completed in {time.time() - start:.2f}s")

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
