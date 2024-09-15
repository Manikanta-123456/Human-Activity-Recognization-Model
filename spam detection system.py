import pandas as pd
import zipfile
import requests
import os

# Define URLs and paths
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
zip_path = 'smsspamcollection.zip'
extract_path = 'smsspamcollection/'

# Download the ZIP file
response = requests.get(url)
with open(zip_path, 'wb') as f:
    f.write(response.content)

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load the dataset
file_path = os.path.join(extract_path, 'SMSSpamCollection')
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])

# Preprocess dataset
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X = df['message']
y = df['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to feature vectors
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Predict and evaluate
from sklearn import metrics
y_pred = model.predict(X_test_counts)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
report = metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print('Classification Report:')
print(report)
