# 🧠 Human Activity Recognition Model

This project uses a **Random Forest Classifier** to recognize human activities based on smartphone sensor data from the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

It classifies activities like walking, sitting, and lying down using accelerometer and gyroscope readings collected from smartphones.

---

## 📁 Dataset

We use the **UCI HAR Dataset**, which includes:

- Measurements from 30 participants
- Sensor signals from a smartphone's accelerometer and gyroscope
- Pre-processed and normalized training and test data

📂 Folder structure:

---

## 🧪 Activities Classified

The following 6 human activities are recognized:

1. 🚶 WALKING  
2. 🧗‍♂️ WALKING_UPSTAIRS  
3. 🧎 WALKING_DOWNSTAIRS  
4. 🪑 SITTING  
5. 🧍 STANDING  
6. 🛌 LAYING  

---

## ⚙️ Technology Stack

- **Language**: Python 3.10  
- **Libraries**:
  - `pandas`
  - `scikit-learn`
  - `matplotlib` (optional for plots)

- **Algorithm**: `RandomForestClassifier` from scikit-learn

---

## 🚀 How to Run the Project

1. 📥 Clone the repository

```bash
git clone https://github.com/Manikanta-123456/Human-Activity-Recognization-Model.git
cd Human-Activity-Recognization-Model
Human-Activity-Recognization-Model/
├── activity_recognition.py
├── environment.yml
└── UCI HAR Dataset/
python activity_recognition.py
Accuracy: 93.58%
Actual: WALKING, Predicted: WALKING
Actual: SITTING, Predicted: STANDING
Actual: STANDING, Predicted: STANDING
...
conda env create -f environment.yml
conda activate har-env
👤 Author
Manikanta Pilli
📧 Email: pillimanikanta44@gmail.com

---

