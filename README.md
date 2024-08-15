# TextClassification_SpamFilter_NaiveBayes

This project demonstrates how to build a spam/not spam text classification model using Python. The model is trained using the Naive Bayes algorithm and evaluated for its performance. This README provides an overview of the project, including setup instructions, data processing, model training, and prediction steps.

## Table of Contents

1. [Setup](#setup)
2. [Data Loading](#data-loading)
3. [Library Imports](#library-imports)
4. [Data Processing](#data-processing)
5. [Train/Test Split](#traintest-split)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Prediction](#prediction)
9. [Usage](#usage)

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/spam-classification.git
   cd spam-classification
   ```
2. **Install Dependencies**
Ensure you have Python 3.x installed. Install the required libraries using pip:

   ```python
   pip install pandas numpy matplotlib nltk scikit-learn gdown
   ```

## Data Loading
The dataset is expected to be a CSV file with two columns: Message and Category. Place the dataset in the appropriate directory or modify the dataset path as needed.

  ```python
  from google.colab import drive
  
  drive.mount('/content/drive', force_remount=True)
  !cp /path/to/dataset/on/your/drive.
    
  !gdown --id 1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R
```


## Library Imports
Import the necessary libraries to process data, train the model, and evaluate performance:

  ```bash
    import string
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
  ```

## Data Processing

### Feature Processing

Preprocess the text data to prepare it for model training:

- **Lowercasing**: Convert text to lowercase.
- **Punctuation Removal**: Remove punctuation from text.
- **Tokenization**: Split text into tokens (words).
- **Stopwords Removal**: Remove common words that do not add meaning.
- **Stemming**: Reduce words to their base or root form.

Here is the code for each of these preprocessing steps:

```python
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def lowercase(text):
    """Convert text to lowercase."""
    return text.lower()

def punctuation_removal(text):
    """Remove punctuation from text."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tokenize(text):
    """Split text into tokens (words)."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    """Remove common words that do not add meaning."""
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    """Reduce words to their base or root form."""
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    """Apply all preprocessing steps to the text."""
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens
```

## Train/Test Split

Split the dataset into training, validation, and test sets:

```python
from sklearn.model_selection import train_test_split

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)

# Further split the training set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED)
```
## Model Training

Train the Naive Bayes classifier using the training data:

```python
from sklearn.naive_bayes import GaussianNB

# Initialize and train the model
model = GaussianNB()
print('Start training...')
model = model.fit(X_train, y_train)
print('Training completed!')
```

## Model Evaluation

Evaluate the model's performance on validation and test sets:

```python
from sklearn.metrics import accuracy_score

# Predict and evaluate on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Val accuracy: {val_accuracy}')

# Predict and evaluate on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test accuracy: {test_accuracy}')
```

## Prediction

Use the trained model to predict whether a new message is spam or not:

```python
def predict(text, model, dictionary):
    """Predict if the given text is spam or not."""
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls

# Example prediction
test_input = 'Hong Kiet is a handsome'
prediction_cls = predict(test_input, model, dictionary)
print(f'Prediction: {prediction_cls}')
```
## Usage

1. **Update Dataset Path**: Ensure that the path to the dataset in the code matches the location where you have saved it. You may need to adjust the `DATASET_PATH` variable in the script to point to your dataset file.

2. **Run the Script**: Execute the Python script in your environment. This will perform the following steps:
   - Load the dataset.
   - Preprocess the data.
   - Split the data into training, validation, and test sets.
   - Train the Naive Bayes model.
   - Evaluate the model's performance.
   - Make predictions.
  
# Additional Information

For any questions or issues, please open an issue in the repository or contact me at [truonghongkietcute@gmail.com](mailto:truonghongkietcute@gmail.com).

Feel free to customize the project names, descriptions, and any other details specific to your projects. If you encounter any problems or have suggestions for improvements, don't hesitate to reach out. Your feedback and contributions are welcome!

Let me know if there’s anything else you need or if you have any other questions. I’m here to help!


