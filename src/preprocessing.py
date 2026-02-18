from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import numpy as np
import tensorflow as tf

PAD = "<pad>"
UNK = "<unk>"


def clean_text(text: str) -> str:
    """
    Normalizes the input text by removing s[ecial characters, punctuation and converting to lowercase.
    
    :param text: The input text to normalize.
    :return: The normalized text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes the input text into a list of words.
    
    :param text: The input text to tokenize.
    :return: A list of tokens (words).
    """
    return text.split()

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by tokenizing the 'description' column and creating a new 'tokens' column.
    
    :param df: The input DataFrame with a 'description' column.
    :return: A new DataFrame with an additional 'tokens' column.
    """
    # Clean the 'description' column
    df['description'] = df['description'].apply(clean_text)
    # Tokenize the cleaned 'description' column
    df['tokens'] = df['description'].apply(tokenize_text)
    return df



def feature_engineering_tf_text_vectorization(
    dataset: pd.DataFrame,
    column_name: str,
    *,
    max_tokens: int = 20000,
    output_sequence_length: int = 128,
    vectorizer: Any | None = None,
)-> tuple[np.ndarray, Any]:
    """
    Convert raw text into fixed-length integer token-id sequences using Keras TextVectorization.

    :param dataset: The input DataFrame containing the text data.
    :param column_name: The name of the column in the DataFrame that contains the text
    :param max_tokens: The maximum number of tokens to keep in the vocabulary (default 20,000)
    :param output_sequence_length: The fixed length of the output token sequences (default 128)
    :param vectorizer: An optional pre-fitted TextVectorization layer to use for transforming the text. If None, a new one will be created and fitted on the dataset.
    :return: A tuple containing the transformed token sequences as a NumPy array and the fitted
    """
    text_data = dataset[column_name].astype(str).values

    if vectorizer is None:
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=output_sequence_length,
            standardize="lower_and_strip_punctuation",
        )
        vectorizer.adapt(text_data)
        # Convert TensorFlow tensor to NumPy array
        X = vectorizer(text_data).numpy()
        return X, vectorizer

    # Convert TensorFlow tensor to NumPy array
    X = vectorizer(text_data).numpy()
    return X