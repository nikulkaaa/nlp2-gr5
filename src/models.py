from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Any
import numpy as np
import pandas as pd
import tensorflow as tf

if TYPE_CHECKING:
    PandasSeriesAny: TypeAlias = pd.Series[Any]
else:
    PandasSeriesAny: TypeAlias = pd.Series

def build_text_classification_cnn(vocab_size: int, embed_dim: int, num_classes: int) -> tf.keras.Sequential:
    """
    Build a Convolutional Neural Network (CNN) model for text classification.
    
    :return: A compiled Keras Sequential model ready for training.
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_size: int, embed_dim: int, num_classes: int) -> tf.keras.Sequential:
    """
    Build a Long Short-Term Memory (LSTM) model for text classification.
    
    :return: A compiled Keras Sequential model ready for training.
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    
def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: PandasSeriesAny | np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: PandasSeriesAny | np.ndarray | None = None,
    vocab_size: int = 20000, 
    embed_dim: int = 128,
    num_classes: int = 4,
    epochs: int = 2,
    batch_size: int = 128
) -> tf.keras.Sequential:
    """
    Train a machine learning model based on the specified model name.

    :param model_name: The name of the model to train. Supported values are 'cnn' and 'lstm'.
    :param X_train: The training data features as a NumPy array.
    :param y_train: The training data labels.
    :param X_val: The validation data features as a NumPy array (optional).
    :param y_val: The validation data labels (optional).
    :param vocab_size: Size of the vocabulary (default: 20000).
    :param embed_dim: Dimensionality of word embeddings (default: 128).
    :param num_classes: Number of output classes (default: 4).
    :param epochs: Number of training epochs (default: 2).
    :param batch_size: Number of samples per gradient update (default: 128).
    :raise ValueError: If an unsupported model name is provided.
    :return: The trained model instance.
    """
    if model_name == 'cnn':
        model = build_text_classification_cnn(vocab_size, embed_dim, num_classes)
        print("Initialized CNN model...")
    elif model_name == 'lstm':
        model = build_lstm_model(vocab_size, embed_dim, num_classes)
        print("Initialized LSTM model...")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
    print(f"Training with batch_size={batch_size}, epochs={epochs}, validation_data={'provided' if validation_data else 'none'}")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1)
    return model

