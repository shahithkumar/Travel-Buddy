import pandas as pd
from recommendation import train_and_save_model

# Path to your dataset (ensure this file exists in the folder)
data_file = 'repeated_travel_destinations.csv'  # Replace with the path to your dataset

# Path to save the trained model and label encoders
model_file = 'rf_model.pkl'
encoders_file = 'label_encoders.pkl'

# Train the model and save it
train_and_save_model(data_file, model_file, encoders_file)  # Pass encoders_file to the function

print("Model and label encoders saved successfully!")
