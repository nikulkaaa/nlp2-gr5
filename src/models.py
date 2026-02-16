from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    PandasSeriesAny: TypeAlias = pd.Series[Any]
else:
    PandasSeriesAny: TypeAlias = pd.Series

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

def evaluate_model(
    model: LogisticRegression | LinearSVC,
    X_test: np.ndarray,
    y_test: PandasSeriesAny | np.ndarray,
) -> dict[str, float]:
    """
    Evaluate the performance of a trained model on the test dataset.
    
    :param model: The trained machine learning model to evaluate.
    :param X_test: The test data features as a NumPy array.
    :param y_test: The test data labels.
    :return: A dictionary containing the accuracy and macro F1 score of the model on the test dataset.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }
    return metrics

def collect_misclassified_samples(
    model: LogisticRegression | LinearSVC,
    X_test: np.ndarray,
    y_test: PandasSeriesAny | np.ndarray,
    *,
    test_df: pd.DataFrame | None = None,
    text_column: str = "description",
    n_samples: int = 20,
    random_state: int = 1337,
) -> pd.DataFrame:
    """
    Collect misclassified samples from the test dataset.

    :param model: The trained machine learning model to evaluate.
    :param X_test: The test data features as a NumPy array.
    :param y_test: The test data labels.
    :param test_df: Optional DataFrame aligned with X_test/y_test; when provided, includes text for inspection.
    :param text_column: Column in test_df that contains the raw text (defaults to "description").
    :param n_samples: Number of misclassified examples to return (default 20). If fewer exist, returns all.
    :param random_state: Seed used for sampling.
    :return: A DataFrame containing misclassified samples with labels and optional text context.
    """
    # Predict labels for the test set
    y_pred = model.predict(X_test)

    # Convert y_test to a NumPy array if it's a Pandas Series, ensuring alignment with y_pred
    y_true = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)

    # Identify indices of misclassified samples
    misclassified_indices = np.flatnonzero(y_pred != y_true)

    # If no misclassified samples, return an empty DataFrame with the expected columns
    if len(misclassified_indices) == 0:
        return pd.DataFrame(columns=["pos", "true_label", "predicted_label"])

    # If n_samples is specified and there are more misclassified samples than n_samples, randomly sample n_samples indices
    if n_samples is not None and n_samples > 0 and len(misclassified_indices) > n_samples:
        rng = np.random.default_rng(random_state)
        misclassified_indices = rng.choice(misclassified_indices, size=n_samples, replace=False)
        misclassified_indices.sort()

    # Create a DataFrame to store the misclassified samples
    result = pd.DataFrame(
        {
            "pos": misclassified_indices,
            "true_label": y_true[misclassified_indices],
            "predicted_label": y_pred[misclassified_indices],
        }
    )

    # If a test_df is provided, include the corresponding text for the misclassified samples. Otherwise, keep numeric features.
    if test_df is not None:
        if len(test_df) != len(X_test):
            raise ValueError(
                "test_df must be aligned with X_test/y_test (same number of rows). "
                f"Got len(test_df)={len(test_df)} and len(X_test)={len(X_test)}"
            )
        if text_column not in test_df.columns:
            raise ValueError(f"test_df does not contain column '{text_column}'")

        result["row_index"] = test_df.index.to_numpy()[misclassified_indices]
        result[text_column] = test_df[text_column].to_numpy()[misclassified_indices]
    else:
        # Keep numeric features available when no raw-text DataFrame is provided.
        result["features"] = list(X_test[misclassified_indices])

    return result