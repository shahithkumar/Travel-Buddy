from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from datetime import date
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200), nullable=False)
    date = db.Column(db.Date, nullable=False)
    user = db.relationship('User', backref=db.backref('expenses', lazy=True))

# Load and preprocess the datasets
destination_file_path = "C:\\Users\\shahi\\Downloads\\destination_dataset.csv"  # Dataset for destinations
user_file_path = "C:\\Users\\shahi\\Downloads\\user_dataset.csv"  # Dataset for user matching

# Load datasets
destination_data = pd.read_csv(destination_file_path)
user_data = pd.read_csv(user_file_path)

# Preprocessing for destination recommendation model
destination_label_encoders = {}
for column in destination_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    destination_data[column] = le.fit_transform(destination_data[column])
    destination_label_encoders[column] = le

destination_features = destination_data.drop(columns=['Destination'])
destination_target = destination_data['Destination']

destination_scaler = StandardScaler()
scaled_destination_features = destination_scaler.fit_transform(destination_features)

# Train the Random Forest model for destination recommendation
destination_rf_model = RandomForestClassifier(random_state=42)
destination_rf_model.fit(scaled_destination_features, destination_target)

# Preprocessing for user matching model
user_label_encoders = {}
for column in user_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    user_data[column] = le.fit_transform(user_data[column])
    user_label_encoders[column] = le

user_features = user_data.drop(columns=['Name', 'Contact', 'Gender'])
user_target = user_data[['Name', 'Contact', 'Gender']]  # We use Name, Contact, Gender as target info

user_scaler = StandardScaler()
scaled_user_features = user_scaler.fit_transform(user_features)

# Functions for ML processing
def recommend_destination(input_features, model, encoders, destination_encoder):
    """Predict a travel destination based on input features."""
    encoded_features = {}
    for key, value in input_features.items():
        if key in encoders:
            encoded_features[key] = encoders[key].transform([value])[0]
        else:
            encoded_features[key] = value
    input_df = pd.DataFrame([encoded_features])
    scaled_input = destination_scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return destination_encoder.inverse_transform(prediction)[0]

def find_best_match(input_features, all_features, encoders, scaler, normalized_features, full_data):
    """Find the best user match based on input features."""
    encoded_features = {}
    for key, value in input_features.items():
        if key in encoders:
            encoded_features[key] = encoders[key].transform([value])[0]
        else:
            encoded_features[key] = value
    input_df = pd.DataFrame([encoded_features])
    scaled_input = scaler.transform(input_df)
    similarities = cosine_similarity(scaled_input, normalized_features)
    best_match_index = np.argmax(similarities)
    return full_data.iloc[best_match_index][['Name', 'Contact', 'Gender']].to_dict()

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            return 'Invalid login credentials'
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            return 'User already exists'
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/destination', methods=['GET', 'POST'])
def destination_recommendation():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        input_features = {
            "Destination Type": request.form['destination_type'],
            "Budget": request.form['budget'],
            "Travel Time": request.form['travel_time'],
            "Activities": request.form.getlist('activities'),
            "Main Goal": request.form['goal'],
            "Environment": request.form['environment'],
            "Weather": request.form['weather'],
            "Food Preferences": request.form['food_preferences'],
            "Region": request.form['region']
        }
        predicted_destination = recommend_destination(
            input_features, destination_rf_model, destination_label_encoders, destination_label_encoders['Destination']
        )
        return render_template("destination_result.html", destination=predicted_destination)
    return render_template("destination.html")

@app.route('/find_match', methods=['POST'])
def find_match():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    input_features = {
        "Destination Type": request.form['destination_type'],
        "Budget": request.form['budget'],
        "Travel Time": request.form['travel_time'],
        "Activities": request.form.getlist('activities'),
        "Main Goal": request.form['goal'],
        "Environment": request.form['environment'],
        "Weather": request.form['weather'],
        "Food Preferences": request.form['food_preferences'],
        "Region": request.form['region']
    }
    user_match = find_best_match(
        input_features, user_features, user_label_encoders, user_scaler, scaled_user_features, user_data
    )
    return render_template("final_result.html", user_match=user_match)

@app.route('/expense_tracking', methods=['GET', 'POST'])
def expense_tracking():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    user = User.query.get(user_id)
    if request.method == 'POST':
        amount = float(request.form['amount'])
        description = request.form['description']
        expense_date = date.fromisoformat(request.form['date'])
        new_expense = Expense(user_id=user.id, amount=amount, description=description, date=expense_date)
        db.session.add(new_expense)
        db.session.commit()
    expenses = Expense.query.filter_by(user_id=user.id).all()
    return render_template('expense_tracking.html', expenses=expenses)

@app.route('/final_result', methods=['GET'])
def final_result():
    return render_template('final_result.html')

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
