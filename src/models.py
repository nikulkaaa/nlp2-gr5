from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    PandasSeriesAny: TypeAlias = pd.Series[Any]
else:
    PandasSeriesAny: TypeAlias = pd.Series

def define_text_classification_cnn():
    """Define a simple CNN architecture for text classification."""
    torch.manual_seed(1337)  # For reproducibility
    vocab_size = 10000  # Size of the vocabulary    
    embedding_dim = 100  # Dimension of word embeddings

    # Define the model architecture
    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab_size, embedding_dim),
        torch.nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(kernel_size=2),
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 48, 4)  # Assuming input sequences of length 100
    )
def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: PandasSeriesAny | np.ndarray,
) -> LogisticRegression | LinearSVC:
    """
    Train a machine learning model based on the specified model name.

    :param model_name: The name of the model to train. Supported values are 'logistic_regression' and 'linear_svc'.
    :param X_train: The training data features as a NumPy array.
    :param y_train: The training data labels.
    :raise ValueError: If an unsupported model name is provided.
    :return: The trained model instance.
    """
    if model_name == 'logistic_regression':
        model = LogisticRegression()
    elif model_name == 'linear_svm':
        model = LinearSVC()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model.fit(X_train, y_train)
    return model

