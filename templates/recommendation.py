import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load Dataset
data = pd.read_csv("C:\Users\shahi\OneDrive\Desktop\New folder\myenv\my_flask_project\repeated_travel_destinations.csv")

# Preprocess Data
label_encoders = {}
for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Feature and Target Split
X = data.drop(columns=["Destination"])
y = data["Destination"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save Model and Encoders
with open("../models/rf_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open("../models/label_encoders.pkl", "wb") as enc_file:
    pickle.dump(label_encoders, enc_file)

print("Random Forest model and encoders saved successfully!")
