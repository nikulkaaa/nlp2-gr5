from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data, feature_engineering
from src.models import train_model
from src.evaluation import evaluate_model, collect_misclassified_samples, plot_confusion_matrix
import json

class Pipeline:
    """ A class to encapsulate the entire machine learning pipeline for the AG News classification task, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples."""
    def __init__(self) -> None:
        """
        Initialize the Pipeline class with placeholders for datasets, models, and evaluation results.

        :return: None
        """
        self.train = None
        self.dev = None
        self.test = None
        self.CNN = None
        self.LSTM = None

    def run(self) -> None:
        """
        Execute the entire machine learning pipeline, including data loading, preprocessing, model training, evaluation, and analysis of misclassified samples.

        :return: None
        """
        # Load data
        self.train, self.test = load_data()
        
        # Split dataset
        self.train, self.dev = split_dataset(self.train)
        
        # Preprocess data
        self.train = preprocess_data(self.train)
        self.dev = preprocess_data(self.dev)
        self.test = preprocess_data(self.test)

        # Maximum number of tokens to consider in the vocabulary
        max_tokens = 10000  # Reduced from 20000 for faster training
        # Maximum length of the output sequence after vectorization (padding/truncating)
        output_sequence_length = 128
        # Dimensionality of the embedding layer
        embed_dim = 64

        # Feature engineering using TF Text Vectorization
        self.X_train, vocab = feature_engineering(self.train, column_name="description", max_tokens=max_tokens, output_sequence_length=output_sequence_length)
         # Convert from 1-indexed to 0-indexed
        self.y_train = self.train['label'].values - 1

        self.X_dev, _ = feature_engineering(self.dev, column_name="description", max_tokens=max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_dev = self.dev['label'].values - 1

        self.X_test, _ = feature_engineering(self.test, column_name="description", max_tokens=max_tokens, output_sequence_length=output_sequence_length, vocab=vocab)
        self.y_test = self.test['label'].values - 1

        print("First 5 rows of the training set after feature engineering: \n", self.X_train[:5])
        print(f"\nTraining on {len(self.X_train)} samples with validation on {len(self.X_dev)} samples...")

        # Train CNN model with larger batch size for faster training, using dev set for validation
        self.CNN = train_model('cnn', self.X_train, self.y_train, X_val=self.X_dev, y_val=self.y_dev, vocab_size=max_tokens, embed_dim=embed_dim, epochs=1, batch_size=256)
        # Train LSTM model with larger batch size for faster training, using dev set for validation
        self.LSTM = train_model('lstm', self.X_train, self.y_train, X_val=self.X_dev, y_val=self.y_dev, vocab_size=max_tokens, embed_dim=embed_dim, epochs=1, batch_size=256)

        print("CNN and LSTM models trained successfully.")
        # # Evaluate models on the test set
        # self.CNN_predictions, self.CNN_metrics = evaluate_model(self.CNN, self.X_test, self.y_test)
        # self.LSTM_predictions, self.LSTM_metrics = evaluate_model(self.LSTM, self.X_test, self.y_test)

        # # Read which model performed better based on the macro_f1 metric (dont think this is necessary)
        # # self.best_model = self.logistic_regression if self.lr_metrics['macro_f1'] > self.svm_metrics['macro_f1'] else self.svm
        # # self.best_model_name = "lr" if self.lr_metrics['macro_f1'] > self.svm_metrics['macro_f1'] else "svm"

        # # Collect misclassified samples on the best performing model
        # # self.best_misclassified = collect_misclassified_samples(self.best_model, self.X_test, self.y_test, n_samples =10)

        # # Collect misclassified for both models for creation of error categories
        # self.CNN_misclassified = collect_misclassified_samples(self.CNN, self.X_test, self.y_test, n_samples=10)
        # self.LSTM_misclassified = collect_misclassified_samples(self.LSTM, self.X_test, self.y_test, n_samples=10)

        # self.predictions = {
        #     "CNN": pipeline.CNN_predictions,
        #     "LSTM": pipeline.LSTM_predictions
        # }
        
        # for model_name, y_pred in self.predictions.items():
        #     plot_confusion_matrix(
        #         pipeline.y_test, 
        #         y_pred, 
        #         f"Confusion Matrix – {model_name}"
        #     )


if __name__ == "__main__":
    # Instantiate and run the machine learning pipeline for AG News classification
    pipeline = Pipeline()
    pipeline.run()

    # # Print evaluation metrics for both models and save them to JSON files, along with the misclassified samples for further analysis.
    # print("CNN Metrics:", pipeline.CNN_metrics)
    # print("LSTM Metrics:", pipeline.LSTM_metrics)

    # # Save metrics and misclassified samples to files for further analysis and reporting.
    # with open('results/cnn_metrics.json', 'w') as f:
    #     json.dump(pipeline.CNN_metrics, f, indent=4)

    # with open('results/lstm_metrics.json', 'w') as f:
    #     json.dump(pipeline.LSTM_metrics, f, indent=4)

    # # Save misclassified samples for the best performing model and both models for error analysis.
    # pipeline.best_misclassified.to_csv(f'results/best_model_{pipeline.best_model_name}_misclassified.csv', index=False)
    # pipeline.CNN_misclassified.to_csv('results/CNN_misclassified.csv', index=False)
    # pipeline.LSTM_misclassified.to_csv('results/LSTM_misclassified.csv', index=False)