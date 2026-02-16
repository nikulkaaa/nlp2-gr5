import pandas as pd

splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
df = pd.read_json("hf://datasets/sh0416/ag_news/" + splits["train"], lines=True)
print(df.head())