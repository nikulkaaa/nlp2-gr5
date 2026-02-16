from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data


class Pipeline:
    def __init__(self):
        self.train = None
        self.dev = None
        self.test = None

    def run(self):
        # Load data
        self.train, self.test = load_data()
        
        # Split dataset
        self.train, self.dev = split_dataset(self.train)
        
        # Preprocess data
        self.train = preprocess_data(self.train)
        self.dev = preprocess_data(self.dev)
        self.test = preprocess_data(self.test)

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()