# 🧠 Human Activity Recognition Using Smartphones

This project applies machine learning (Random Forest) to recognize human activities (walking, sitting, etc.) using smartphone sensor data from the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

---

## 📊 Dataset

The dataset contains accelerometer and gyroscope readings from smartphones worn by 30 volunteers during daily activities.

**Activities classified:**
1. WALKING  
2. WALKING_UPSTAIRS  
3. WALKING_DOWNSTAIRS  
4. SITTING  
5. STANDING  
6. LAYING  

📁 Dataset folder:  
`UCI HAR Dataset/train/`  
`UCI HAR Dataset/test/`

---

## 🛠️ Technologies Used

- Python 3.x
- pandas
- scikit-learn
- RandomForestClassifier

---

## 🚀 How to Run

1. 📥 Download the dataset:  
   [https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

2. 📂 Unzip the dataset and place it as:  
   `UCI HAR Dataset/UCI HAR Dataset/train/...`

3. ✅ Run the Python script:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
def load_dataset(filepath):
    df = pd.read_csv(filepath, sep='\s+', header=None)
    return df

# Load the data
def load_data():
    X_train = load_dataset('UCI HAR Dataset/train/X_train.txt')
    y_train = load_dataset('UCI HAR Dataset/train/y_train.txt')
    X_test = load_dataset('UCI HAR Dataset/test/X_test.txt')
    y_test = load_dataset('UCI HAR Dataset/test/y_test.txt')
    return X_train, y_train, X_test, y_test

# Model training and evaluation
X_train, y_train, X_test, y_test = load_data()
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
