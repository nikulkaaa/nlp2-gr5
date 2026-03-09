from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data, feature_engineering
from src.models import train_model
from src.evaluation import evaluate_model, collect_misclassified_samples, plot_confusion_matrix, plot_learning_curves
import json

class Pipeline:
    """ A class to encapsulate the entire machine learning pipeline for the AG News classification task, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples."""
    def __init__(self, max_length:int=128) -> None:
        """
        Initialize the Pipeline class with placeholders for datasets, models, and evaluation results.

        :return: None
        """
        self.train = None
        self.dev = None
        self.test = None
        self.CNN = None
        self.LSTM = None
        self.max_tokens = 10000
        self.max_length = max_length

    def run(self) -> None:
        """
        Execute the entire machine learning pipeline, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples.

        :return: None
        """
        # Load data
        self.train, self.test = load_data()
        
        # Split dataset
        self.train, self.dev = split_dataset(self.train)
        print(f"Loaded data for pipeline {self.max_length}")
        
        # Preprocess data
        self.train = preprocess_data(self.train)
        self.dev = preprocess_data(self.dev)
        self.test = preprocess_data(self.test)

        # Maximum length of the output sequence after vectorization (padding/truncating)
        output_sequence_length = self.max_length
        # Dimensionality of the embedding layer
        embed_dim = 64

        # Number of training epochs for both models
        epochs = 30

        
        # Feature engineering using TF Text Vectorization
        self.X_train, vocab = feature_engineering(self.train, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length)
         # Convert from 1-indexed to 0-indexed
        self.y_train = self.train['label'].values - 1

        self.X_dev, _ = feature_engineering(self.dev, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_dev = self.dev['label'].values - 1

        self.X_test, _ = feature_engineering(self.test, column_name="description", max_tokens=self.max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_test = self.test['label'].values - 1

        print(f"Starting training for pipeline {self.max_length}")
        # Train CNN model with larger batch size for faster training, using dev set for validation
        self.CNN, self.CNN_history = train_model('cnn', self.X_train, self.y_train, X_val=self.X_dev, y_val=self.y_dev, vocab_size=self.max_tokens, embed_dim=embed_dim, epochs=epochs, batch_size=256)
        # Train LSTM model with larger batch size for faster training, using dev set for validation
        self.LSTM, self.LSTM_history = train_model('lstm', self.X_train, self.y_train, X_val=self.X_dev, y_val=self.y_dev, vocab_size=self.max_tokens, embed_dim=embed_dim, epochs=epochs, batch_size=256)

        print(f"Starting evaluation for pipeline {self.max_length}")
        # Evaluate models on the test set
        self.CNN_predictions, self.CNN_metrics = evaluate_model(self.CNN, self.X_test, self.y_test)
        self.LSTM_predictions, self.LSTM_metrics = evaluate_model(self.LSTM, self.X_test, self.y_test)

        # Read which model performed better based on the macro_f1 metric
        self.best_model = self.CNN if self.CNN_metrics['macro_f1'] > self.LSTM_metrics['macro_f1'] else self.LSTM
        self.best_model_name = "CNN" if self.CNN_metrics['macro_f1'] > self.LSTM_metrics['macro_f1'] else "LSTM"

        # Collect misclassified samples on the best performing model
        self.best_misclassified = collect_misclassified_samples(self.best_model, self.X_test, self.y_test, n_samples =10)

        # Collect misclassified for both models for creation of error categories
        self.CNN_misclassified = collect_misclassified_samples(self.CNN, self.X_test, self.y_test, n_samples=10)
        self.LSTM_misclassified = collect_misclassified_samples(self.LSTM, self.X_test, self.y_test, n_samples=10)

        self.predictions = {
            "CNN": self.CNN_predictions,
            "LSTM": self.LSTM_predictions
        }
        
        for model_name, y_pred in self.predictions.items():
            plot_confusion_matrix(
                self.y_test, 
                y_pred, 
                f"Confusion Matrix – {model_name}, Max Length={self.max_length}"
            )
        
        # Plot learning curves for both models
        plot_learning_curves(
            {"CNN": self.CNN_history, "LSTM": self.LSTM_history},
            title=f"Learning Curves – max_length={self.max_length}", 
            max_tokens=self.max_length
        )

def ablation_study(max_length:int) -> None:
    """
    Conduct an ablation study by running the pipeline with different maximum sequence lengths for the text vectorization step, and save the results for analysis.

    :param max_length: The maximum sequence length (number of tokens) after padding/truncating (e.g., 64, 128, 256).
    :return: None
    """
    pipeline = Pipeline(max_length=max_length)
    pipeline.run()
    # Save metrics for ablation study with max_length
    with open(f'results/ablation_max_length_{max_length}.json', 'w') as f:
        json.dump({
            "CNN": pipeline.CNN_metrics,
            "LSTM": pipeline.LSTM_metrics
        }, f, indent=4)

if __name__ == "__main__":
    # Instantiate and run the machine learning pipeline for AG News classification with max_tokens 1000
    pipeline = Pipeline()
    pipeline.run()

    # Print evaluation metrics for both models and save them to JSON files, along with the misclassified samples for further analysis.
    print("CNN Metrics:", pipeline.CNN_metrics)
    print("LSTM Metrics:", pipeline.LSTM_metrics)

    # Save metrics and misclassified samples to files for further analysis and reporting.
    with open('results/cnn_metrics.json', 'w') as f:
        json.dump(pipeline.CNN_metrics, f, indent=4)

    with open('results/lstm_metrics.json', 'w') as f:
        json.dump(pipeline.LSTM_metrics, f, indent=4)
    
    # Save training history for both models
    with open('results/cnn_history.json', 'w') as f:
        json.dump(pipeline.CNN_history, f, indent=4)
    
    with open('results/lstm_history.json', 'w') as f:
        json.dump(pipeline.LSTM_history, f, indent=4)

    # Save misclassified samples for the best performing model and both models for error analysis.
    pipeline.best_misclassified.to_csv(f'results/best_model_{pipeline.best_model_name}_misclassified.csv', index=False)
    pipeline.CNN_misclassified.to_csv('results/CNN_misclassified.csv', index=False)
    pipeline.LSTM_misclassified.to_csv('results/LSTM_misclassified.csv', index=False)

    # Conduct an ablation study by running the pipeline with different maximum sequence lengths for the text vectorization step, and save the results for analysis.
    for max_length in [10, 20, 30, 64, 128]:
        ablation_study(max_length)
    
