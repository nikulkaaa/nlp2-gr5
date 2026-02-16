from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    Evaluate the performance of a trained model on the test dataset.
    
    :param model: The trained machine learning model to evaluate.
    :param X_test: The test data features as a NumPy array.
    :param y_test: The test data labels.
    :return: A dictionary containing the accuracy and macro F1 score of the model on the test dataset.
    """
    y_pred = model.predict(X_test)
    # Convert NumPy scalars/arrays into plain Python types so metrics can be JSON-serialized.
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return y_pred, metrics

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    class_labels = ["World", "Sports", "Business", "Sci/Tech"]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"results/{title}.png")
    plt.close()
    
def collect_misclassified_samples(
    model: LogisticRegression | LinearSVC,
    X_test: np.ndarray,
    y_test: PandasSeriesAny | np.ndarray,
    *,
    test_df: pd.DataFrame | None = None,
    text_column: str = "description",
    n_samples: int = 20,
    random_state: int = 1337,
    include_text: bool = True,
    include_features: bool = False,
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
    :param include_text: When True (default), include the original text column for easier inspection.
                         If test_df is not provided, this function will attempt to load the AG News test split
                         from Hugging Face (`sh0416/ag_news`) and use it as the source of text.
    :param include_features: When True, include numeric feature vectors for the misclassified rows.
                             Defaults to False to avoid writing huge TF-IDF arrays to CSV.
    :return: A DataFrame containing misclassified samples with labels and optional text context.
    """
    # Label map for legibility
    label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

    # Predict labels for the test set
    y_pred = model.predict(X_test)

    # Convert y_test to a NumPy array if it's a Pandas Series, ensuring alignment with y_pred
    y_true = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)

    # Identify indices of misclassified samples
    misclassified_indices = np.flatnonzero(y_pred != y_true)

    # If no misclassified samples, return an empty DataFrame with the expected columns
    if len(misclassified_indices) == 0:
        base_cols = ["pos", "true_label", "predicted_label"]
        if include_text:
            base_cols.append(text_column)
        if include_features:
            base_cols.append("features")
        return pd.DataFrame(columns=base_cols)

    # If n_samples is specified and there are more misclassified samples than n_samples, randomly sample n_samples indices
    if n_samples is not None and n_samples > 0 and len(misclassified_indices) > n_samples:
        rng = np.random.default_rng(random_state)
        misclassified_indices = rng.choice(misclassified_indices, size=n_samples, replace=False)
        misclassified_indices.sort()

    # Create a DataFrame to store the misclassified samples
    result = pd.DataFrame(
        {
            "pos": misclassified_indices,
            "true_label": [label_map[label] for label in y_true[misclassified_indices]],
            "predicted_label": [label_map[label] for label in y_pred[misclassified_indices]],
        }
    )

    # Include text for inspection (preferred).
    if include_text:
        resolved_test_df: pd.DataFrame | None = test_df

        # If the caller didn't provide a DataFrame, try to reconstruct the AG News test split.
        # This keeps the call-site unchanged while avoiding dumping TF-IDF vectors.
        if resolved_test_df is None:
            try:
                from datasets import load_dataset  # type: ignore

                try:
                    ds = load_dataset("sh0416/ag_news")
                    test_ds = ds["test"]
                except Exception:
                    data_files = {"test": "test.jsonl"}
                    ds = load_dataset("json", data_files=data_files, repo_id="sh0416/ag_news")
                    test_ds = ds["test"]

                to_pandas_result = test_ds.to_pandas()
                if isinstance(to_pandas_result, pd.DataFrame):
                    resolved_test_df = to_pandas_result
                else:
                    # Some `datasets` versions type this as an iterator of DataFrames.
                    resolved_test_df = pd.concat(list(to_pandas_result), ignore_index=True)

                if text_column not in resolved_test_df.columns and "text" in resolved_test_df.columns:
                    resolved_test_df.rename(columns={"text": text_column}, inplace=True)
            except Exception:
                resolved_test_df = None

        if resolved_test_df is not None:
            if len(resolved_test_df) != len(X_test):
                raise ValueError(
                    "test_df must be aligned with X_test/y_test (same number of rows). "
                    f"Got len(test_df)={len(resolved_test_df)} and len(X_test)={len(X_test)}"
                )
            if text_column not in resolved_test_df.columns:
                raise ValueError(f"test_df does not contain column '{text_column}'")

            result["row_index"] = resolved_test_df.index.to_numpy()[misclassified_indices]
            result[text_column] = resolved_test_df[text_column].to_numpy()[misclassified_indices]

    # Optionally include numeric feature vectors (off by default).
    if include_features:
        result["features"] = list(X_test[misclassified_indices])

    return result