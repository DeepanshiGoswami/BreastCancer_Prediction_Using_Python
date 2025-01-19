
# Breast Cancer Detection Using Machine Learning

This repository contains a project that implements machine learning models to detect breast cancer based on a labeled dataset of tumor features. The models aim to classify cases as malignant or benign with high accuracy. Various machine learning algorithms are utilized, including Logistic Regression, Decision Trees, and Random Forest, along with essential techniques for data preprocessing, feature scaling, and model evaluation.

## Project Overview

- **Goal**: To develop machine learning models capable of classifying breast cancer cases as benign or malignant with high accuracy.
- **Techniques**: Data preprocessing, feature scaling, label encoding, model evaluation, and performance metrics (accuracy, precision, recall, F1-score, confusion matrix).
- **Key Algorithm**: Random Forest Classifier showed the best performance in terms of classification accuracy.

## Features

- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset, containing 30 features such as radius_mean, texture_mean, perimeter_mean, etc.
- **Target Variable**: Diagnosis (Malignant = 1, Benign = 0)
- **Data Size**: 569 instances with no missing values in the features.
- **Model Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/breast-cancer-detection.git
    cd breast-cancer-detection
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Files

- **`main.py`**: Main script to load the dataset, preprocess it, and train the models.
- **`model.py`**: Contains the machine learning models and evaluation functions.
- **`data_preprocessing.py`**: Handles all preprocessing steps, including feature scaling and label encoding.
- **`visualization.py`**: Generates visualizations like correlation matrices, feature distributions, and class distribution.
- **`requirements.txt`**: Lists the required Python libraries.
- **`dataset.csv`**: The dataset file (Breast Cancer Wisconsin dataset).

## Usage

1. **Run the main script**:
    ```bash
    python main.py
    ```

2. **Evaluate model performance**:
   - The models are evaluated using accuracy, precision, recall, F1-score, and confusion matrix on the test data.
   - The Random Forest model provides the best classification results.

3. **Visualizations**:
   - The script generates visualizations for data analysis and model performance insights.
   - Key visualizations include feature distributions, class distribution, and a correlation matrix heatmap.

## Model Performance

- **Random Forest Classifier**: Achieved an accuracy of 96.7%, making it the best-performing model.
- **Logistic Regression**: Achieved an accuracy of 95.3%.
- **Decision Tree**: Demonstrated moderate performance, with potential overfitting issues.

## Future Work

- **Hyperparameter Optimization**: Use techniques like GridSearchCV to fine-tune model hyperparameters.
- **Deep Learning**: Implement deep learning models such as CNNs for more complex feature extraction.
- **Clinical Integration**: Integrate the model with clinical imaging data (e.g., mammograms) for more accurate predictions.
- **Deployment**: Create a web-based diagnostic tool using frameworks like Flask or Django for real-time predictions.

## Ethical Considerations

- **Privacy and Security**: Ensuring that patient data is protected in compliance with laws such as HIPAA.
- **Bias and Fairness**: Extending the dataset to be more diverse and representative of different populations.
- **Explainable AI**: Using explainable AI techniques to provide transparency and trust in model predictions.

## References

- UCI Machine Learning Repository: Breast Cancer Wisconsin Dataset
- scikit-learn Documentation
- Zhou, J., & Han, J. (2020). *Explainable AI: Interpreting ML Models in Healthcare*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
