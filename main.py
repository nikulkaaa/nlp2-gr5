from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data, feature_engineering_tfidf
from src.models import train_model, evaluate_model, collect_misclassified_samples, plot_confusion_matrix
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
        self.logistic_regression = None
        self.svm = None
        self.best_model = None
        self.best_model_name = None

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

        # Feature engineering using TF-IDF
        self.X_train, vectorizer = feature_engineering_tfidf(self.train, column_name="description")
        self.y_train = self.train['label']

        self.X_dev = feature_engineering_tfidf(self.dev, column_name="description", vectorizer=vectorizer)
        self.y_dev = self.dev['label']

        self.X_test = feature_engineering_tfidf(self.test, column_name="description", vectorizer=vectorizer)
        self.y_test = self.test['label']

        # Train logistic regression model
        self.logistic_regression = train_model('logistic_regression', self.X_train, self.y_train)
        # Train SVM model
        self.svm = train_model('linear_svm', self.X_train, self.y_train)

        # Evaluate models on the test set
        self.lr_predictions, self.lr_metrics = evaluate_model(self.logistic_regression, self.X_test, self.y_test)
        self.svm_predictions, self.svm_metrics = evaluate_model(self.svm, self.X_test, self.y_test)

        # Read which model performed better based on the macro_f1 metric
        self.best_model = self.logistic_regression if self.lr_metrics['macro_f1'] > self.svm_metrics['macro_f1'] else self.svm
        self.best_model_name = "lr" if self.lr_metrics['macro_f1'] > self.svm_metrics['macro_f1'] else "svm"

        # Collect misclassified samples on the best performing model
        self.best_misclassified = collect_misclassified_samples(self.best_model, self.X_test, self.y_test, n_samples =10)

        # Collect misclassified for both models for creation of error categories
        self.lr_misclassified = collect_misclassified_samples(self.logistic_regression, self.X_test, self.y_test, n_samples=20)
        self.svm_misclassified = collect_misclassified_samples(self.svm, self.X_test, self.y_test, n_samples=20)

        self.predictions = {
            "Logistic Regression": pipeline.lr_predictions,
            "Linear SVM": pipeline.svm_predictions
        }
        
        for model_name, y_pred in self.predictions.items():
            plot_confusion_matrix(
                pipeline.y_test, 
                y_pred, 
                f"Confusion Matrix – {model_name}"
            )


if __name__ == "__main__":
    # Instantiate and run the machine learning pipeline for AG News classification
    pipeline = Pipeline()
    pipeline.run()

    # Print evaluation metrics for both models and save them to JSON files, along with the misclassified samples for further analysis.
    print("Logistic Regression Metrics:", pipeline.lr_metrics)
    print("SVM Metrics:", pipeline.svm_metrics)

    # Save metrics and misclassified samples to files for further analysis and reporting.
    with open('results/logistic_regression_metrics.json', 'w') as f:
        json.dump(pipeline.lr_metrics, f, indent=4)

    with open('results/svm_metrics.json', 'w') as f:
        json.dump(pipeline.svm_metrics, f, indent=4)

    # Save misclassified samples for the best performing model and both models for error analysis.
    pipeline.best_misclassified.to_csv(f'results/best_model_{pipeline.best_model_name}_misclassified.csv', index=False)
    pipeline.lr_misclassified.to_csv('results/logistic_regression_misclassified.csv', index=False)
    pipeline.svm_misclassified.to_csv('results/svm_misclassified.csv', index=False)