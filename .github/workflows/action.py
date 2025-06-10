import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
def load_dataset(filepath):
    # Use sep='\s+' instead of delim_whitespace=True
    df = pd.read_csv(filepath, sep='\s+', header=None)
    return df

# Load the data
def load_data():
    # Update file paths with the extra 'UCI HAR Dataset' folder
    X_train = load_dataset('C:/Users/pilli/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt')
    y_train = load_dataset('C:/Users/pilli/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt')
    
    X_test = load_dataset('C:/Users/pilli/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt')
    y_test = load_dataset('C:/Users/pilli/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt')
    
    return X_train, y_train, X_test, y_test

# Load and prepare data
X_train, y_train, X_test, y_test = load_data()

# Reshape the labels
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Mapping the activities
activities = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# Example predictions
for i in range(5):
    print(f"Actual: {activities[y_test[i]]}, Predicted: {activities[y_pred[i]]}")
