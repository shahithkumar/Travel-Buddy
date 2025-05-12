import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'C:\Users\shahi\OneDrive\Desktop\New folder\myenv\my_flask_project\repeated_travel_destinations.csv'
data = pd.read_csv(file_path)

# Preprocessing
data_cleaned = data.drop(columns=["Duration"])
label_encoders = {}
for column in data_cleaned.columns:
    if data_cleaned[column].dtype == 'object':
        le = LabelEncoder()
        data_cleaned[column] = le.fit_transform(data_cleaned[column])
        label_encoders[column] = le

# Prepare data for modeling
X = data_cleaned.drop(columns=["Destination"])
y = data_cleaned["Destination"]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model and encoders as pickle files
with open('travel_rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

# Define recommendation function for predictions
def recommend_destination(input_features, model, feature_encoders, label_encoder):
    """
    Recommend a travel destination based on user inputs.

    Args:
        input_features (dict): Dictionary with feature names as keys and user input as values.
        model: Trained machine learning model.
        feature_encoders (dict): Encoders for categorical features.
        label_encoder: Encoder for the target variable (Destination). 

    Returns:
        str: Predicted destination.
    """
    # Encode input features
    encoded_input = []
    for feature, value in input_features.items():
        if feature in feature_encoders:
            encoded_value = feature_encoders[feature].transform([value])[0]
        else:
            encoded_value = value
        encoded_input.append(encoded_value)

    # Predict the destination
    predicted_class = model.predict([encoded_input])[0]
    destination = label_encoder.inverse_transform([predicted_class])[0]
    return destination

# Example usage (you can replace this with frontend input)
sample_input = {
    "Destination Type": "Mountain",
    "Budget": "Low",
    "Travel Time": "Winter",
    "Activities": "Trekking, Skiing",
    "Main Goal": "Adventure",
    "Environment": "Active",
    "Weather": "Cold",
    "Food Preferences": "North Indian",
    "Region": "North"
}

# Load the model and encoders for prediction
with open('travel_rf_model.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    loaded_label_encoders = pickle.load(le_file)

# Get the predicted destination
predicted_destination = recommend_destination(
    sample_input, loaded_rf_model, loaded_label_encoders, loaded_label_encoders["Destination"]
)
print(predicted_destination)
