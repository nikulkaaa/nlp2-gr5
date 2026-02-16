import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the AG News dataset from Hugging Face and return the train and test splits as pandas DataFrames.

    :return: A tuple containing the train and test DataFrames.
    """
    splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
    train = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)
    test = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["test"], lines=True)
    return train, test

def split_dataset(train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the original train set into a new train set and a dev set.

    :param train: The original train DataFrame.
    :return: A tuple containing the new train and dev DataFrames.
    """

    x_all = train["description"].tolist()
    y_all = train["label"].tolist()

    x_train, x_dev, y_train, y_dev = train_test_split(
        x_all,
        y_all,
        test_size=0.1,
        random_state=1337,
        stratify=y_all,
        shuffle=True,
    )

    new_train = pd.DataFrame({"description": x_train, "label": y_train})
    dev = pd.DataFrame({"description": x_dev, "label": y_dev})

    return new_train, dev

train, test = load_data()
train, dev = split_dataset(train)



print(f"Train size: {len(train)}")
print(f"Dev size: {len(dev)}")
print(f"Test size: {len(test)}")
