from src.data import load_data, split_dataset
from src.preprocessing import preprocess_data, feature_engineering_tfidf
from src.models import train_model, evaluate_model, collect_misclassified_samples


class Pipeline:
    def __init__(self):
        self.train = None
        self.dev = None
        self.test = None
        self.logistic_regression = None
        self.svm = None

    def run(self):
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
        self.lr_metrics = evaluate_model(self.logistic_regression, self.X_test, self.y_test)
        self.svm_metrics = evaluate_model(self.svm, self.X_test, self.y_test)

        # Collect misclassified samples
        self.lr_misclassified = collect_misclassified_samples(self.logistic_regression, self.X_test, self.y_test)
        self.svm_misclassified = collect_misclassified_samples(self.svm, self.X_test, self.y_test)




if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
    # Print the head of the preprocessed train dataset
    print(pipeline.train.head())
    print(pipeline.X_dev[:5])
    print("Logistic Regression Metrics:", pipeline.lr_metrics)
    print("SVM Metrics:", pipeline.svm_metrics)
    pipeline.lr_misclassified.to_csv('logistic_regression_misclassified.csv', index=False)
    pipeline.svm_misclassified.to_csv('svm_misclassified.csv', index=False)