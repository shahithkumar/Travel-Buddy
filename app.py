from flask import Flask, request, render_template, redirect, session, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import jwt
from functools import wraps
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import secrets
from datetime import datetime
import re

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Models (unchanged)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200), nullable=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipient_name = db.Column(db.String(80), nullable=False)
    content = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)

# Load datasets
data = pd.read_csv(r"C:\Users\shahi\OneDrive\Desktop\New folder\myenv\my_flask_project\merged_travel_and_peopleE.csv")
data.columns = data.columns.str.strip()
data['Budget (₹)'] = pd.to_numeric(data['Budget (₹)'].str.replace(',', ''), errors='coerce').fillna(5000).astype(int)

itinerary_data = pd.read_csv(r"C:\Users\shahi\OneDrive\Desktop\New folder\myenv\my_flask_project\repeated_travel_destinations.csv")
itinerary_data.columns = itinerary_data.columns.str.strip()

city_coords = pd.read_csv(r"C:\Users\shahi\OneDrive\Desktop\New folder\myenv\my_flask_project\city_coordinates.csv")
city_coords.set_index('City', inplace=True)

# Recommendation setup (unchanged)
features = ["Type", "Weather", "Budget (₹)", "Vibe", "Travel Goal"]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(data[features])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(encoded_features)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(scaled_features, data["Destination"])

destination_images = {
    "Agra": "Agra.jpeg", "Andaman": "ANDAMAN.jpg", "Cherrapunji": "Cherrapunjii.jpeg", "Coorg": "Coorg.jpeg",
    "Darjeeling": "Darjeeling.webp", "Goa": "GOA.jpg", "Gokarna": "Gokarna.jpeg", "Gulmarg": "GULMARG.jpeg",
    "Haridwar": "Haridwarr.jpeg", "Jaipur": "Jaipur.jpeg", "Jaisalmer": "Jaisalmer.jpeg", "Kedarnath": "kedarnathh.jpeg",
    "Kovalam": "kovalam.jpeg", "Ladakh": "Ladakh.jpeg", "Lakshadweep": "lakshadweep.jpeg", "Mahabaleshwar": "Mahabaleshwar.jpg",
    "Manali": "Manali.jpeg", "Mount Abu": "Mount Abuu.jpg", "Munnar": "munnar.jpeg", "Ooty": "OOTY.jpeg",
    "Pondicherry": "pondi.jpeg", "Rameswaram": "rameswaram.jpg", "Ranikhet": "Ranikhett.jpg", "Rishikesh": "rishikesh.webp",
    "Shimla": "SHIMLA.jpeg", "Spiti Valley": "Spiti Valleyy.jpg", "Sundarbans": "Sundarbanss.jpeg", "Udaipur": "Udaipurr.jpeg",
    "Varanasi": "Varanasii.jpeg", "Vizag": "Vizag.jpg"
}

# Helper Functions (unchanged except get_dynamic_itinerary)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def parse_cost(cost_str):
    if isinstance(cost_str, (int, float)):
        return int(cost_str)
    if not isinstance(cost_str, str):
        return 0
    return int(re.sub(r'[₹,]', '', cost_str))

def extract_activity_costs(activity_str):
    if not isinstance(activity_str, str):
        return 0
    cost_pattern = r'₹([\d,]+)|Free'
    costs = re.findall(cost_pattern, activity_str)
    total = 0
    for cost in costs:
        if cost and cost != 'Free':
            total += int(cost.replace(',', ''))
    return total

def get_dynamic_itinerary(destination, days, preferences):
    dest_itinerary = itinerary_data[itinerary_data["Destination"].str.strip().str.lower() == destination.strip().lower()]
    max_days = int(dest_itinerary['Day'].max()) if not dest_itinerary.empty else 6  # Default to 6 if no data
    
    days = min(days, max_days) if max_days > 0 else days
    
    if not dest_itinerary.empty:
        itinerary = dest_itinerary.head(days).to_dict(orient='records')
        if len(itinerary) < days:
            repeated = dest_itinerary.to_dict(orient='records')
            while len(itinerary) < days:
                for entry in repeated:
                    if len(itinerary) < days:
                        itinerary.append(entry)
    else:
        itinerary = [
            {
                'Day': i + 1,
                'Morning Activity (Cost)': f"Explore {destination} (₹500)",
                'Afternoon Activity (Cost)': f"Sightseeing in {destination} (₹700)",
                'Evening Activity (Cost)': f"Relax in {destination} (₹600)",
                'Daily Cost (₹)': 1800
            } for i in range(days)
        ]
    
    vibe = preferences.get('Vibe', 'Relaxing').lower()
    goal = preferences.get('Travel Goal', 'Adventure').lower()
    adjusted_itinerary = []
    for day in itinerary:
        adjusted_day = day.copy()
        if 'adventure' in goal:
            for time_slot in ['Morning Activity (Cost)', 'Afternoon Activity (Cost)', 'Evening Activity (Cost)']:
                if time_slot in adjusted_day and adjusted_day[time_slot]:
                    try:
                        activity, cost_str = adjusted_day[time_slot].rsplit(' (₹', 1)
                        cost = int(cost_str.strip(')')) + 200
                        adjusted_day[time_slot] = f"Adventure {activity} (₹{cost})"
                    except (ValueError, IndexError):
                        adjusted_day[time_slot] = f"Adventure {adjusted_day[time_slot].split(' (₹')[0]} (₹500)"
        elif 'relaxing' in vibe:
            for time_slot in ['Morning Activity (Cost)', 'Afternoon Activity (Cost)', 'Evening Activity (Cost)']:
                if time_slot in adjusted_day and adjusted_day[time_slot]:
                    try:
                        activity, cost_str = adjusted_day[time_slot].rsplit(' (₹', 1)
                        cost = max(int(cost_str.strip(')')) - 100, 0)
                        adjusted_day[time_slot] = f"Relax at {activity} (₹{cost})"
                    except (ValueError, IndexError):
                        adjusted_day[time_slot] = f"Relax at {adjusted_day[time_slot].split(' (₹')[0]} (₹400)"
        if 'Daily Cost (₹)' in adjusted_day:
            daily_cost = sum(extract_activity_costs(adjusted_day[slot]) for slot in ['Morning Activity (Cost)', 'Afternoon Activity (Cost)', 'Evening Activity (Cost)'])
            adjusted_day['Daily Cost (₹)'] = daily_cost
        adjusted_itinerary.append(adjusted_day)
    
    adjusted_itinerary = [
        {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v 
         for k, v in day.items()}
        for day in adjusted_itinerary
    ]
    return adjusted_itinerary, max_days

def get_travel_costs(starting_city, destination, days):
    try:
        start_lat = city_coords.loc[starting_city, 'Latitude']
        start_lon = city_coords.loc[starting_city, 'Longitude']
        dest_lat = city_coords.loc[destination, 'Latitude']
        dest_lon = city_coords.loc[destination, 'Longitude']
        distance = haversine_distance(start_lat, start_lon, dest_lat, dest_lon)
        road_distance = distance * 1.2
        
        flight_cost = 2500 + (distance * 6)
        flight_time = round(distance / 800 + 2, 1)
        
        train_time = round(distance / 100, 1)
        train_costs = {
            'sleeper': {'cost': round(distance * 0.8 + 400), 'time': train_time, 'departure': '8:30 PM', 'arrival': '6:45 AM'},
            'ac3': {'cost': round(distance * 1.6 + 800), 'time': train_time - 0.5, 'departure': '9:00 PM', 'arrival': '7:00 AM'},
            'ac2': {'cost': round(distance * 2.4 + 1200), 'time': train_time - 1, 'departure': '8:00 PM', 'arrival': '6:30 AM'}
        }
        
        days_road = max(1, round(road_distance / 600))
        road_cost = round((road_distance * 12) + 1500 + (500 * days_road))
        road_time = round(road_distance / 60, 1)
        google_maps_url = f"https://www.google.com/maps/dir/{starting_city}/{destination}"
        
        is_island = destination in ["Andaman", "Lakshadweep"]
        return {
            'distance': round(distance),
            'road_distance': round(road_distance),
            'flight': {'cost': round(flight_cost * (1.5 if is_island else 1)), 'time': flight_time},
            'train': None if is_island else train_costs,
            'road': None if is_island else {'cost': road_cost, 'time': road_time, 'maps_url': google_maps_url},
            'days': days_road
        }
    except KeyError as e:
        print(f"Coordinates not found for {e}")
        return None

def get_accommodation(destination, days, budget_level='mid-range'):
    base_costs = {'budget': 800, 'mid-range': 2000, 'luxury': 5000}
    cost_per_night = base_costs.get(budget_level, 2000)
    types = {'budget': 'Guesthouse', 'mid-range': '3-Star Hotel', 'luxury': '5-Star Resort'}
    return {
        'type': types.get(budget_level, 'Hotel'),
        'cost_per_night': cost_per_night,
        'total_cost': cost_per_night * days,
        'rating': '4.0/5'
    }

def get_destination_budget(destination):
    dest_data = data[data["Destination"].str.strip().str.lower() == destination.strip().lower()]
    if not dest_data.empty:
        return int(dest_data['Budget (₹)'].mean())  # Average budget for the destination
    return 5000  # Default if no data

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get('token')
        if not token:
            return redirect('/login?next=' + request.path)
        try:
            jwt.decode(token, 'mysecretkey123', algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return redirect('/login?next=' + request.path)
        return f(*args, **kwargs)
    return decorated

# Routes (unchanged until /plan_trip)
@app.route('/')
def index():
    return render_template('homee.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            token = jwt.encode({'user': username}, 'mysecretkey123', algorithm='HS256')
            session['token'] = token
            next_url = request.form.get('next', request.args.get('next', '/'))
            return redirect(next_url)
        return render_template('loginn.html', error="Invalid credentials", next=request.args.get('next', '/'))
    return render_template('loginn.html', next=request.args.get('next', '/'))

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('registerr.html', error="Username and password required")
        hashed_password = generate_password_hash(password)
        if User.query.filter_by(username=username).first():
            return render_template('registerr.html', error="Username exists")
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('registerr.html')

@app.route('/home')
def home():
    return render_template('homee.html')

@app.route('/choose', methods=['GET', 'POST'])
@token_required
def choose():
    if request.method == 'POST':
        session['option'] = request.form.get('option')
        return redirect('/query_form')
    return redirect('/home')

@app.route('/query_form')
@token_required
def query_form():
    return render_template('queryy.html')

@app.route('/recommend', methods=['POST'])
@token_required
def recommend():
    user_input = request.form
    input_df = pd.DataFrame([user_input])
    input_encoded = encoder.transform(input_df[features])
    input_scaled = scaler.transform(input_encoded)
    option = session.get('option', 'destination')
    direction = user_input.get('Direction')
    direction_destinations = {
        'North': ['Manali', 'Shimla', 'Gulmarg', 'Darjeeling', 'Spiti Valley', 'Mount Abu', 'Ranikhet', 'Jaisalmer', 'Ladakh', 'Agra', 'Jaipur', 'Udaipur', 'Varanasi', 'Kedarnath', 'Haridwar', 'Rishikesh'],
        'East': ['Cherrapunji', 'Sundarbans'],
        'South': ['Goa', 'Andaman', 'Kovalam', 'Lakshadweep', 'Pondicherry', 'Vizag', 'Coorg', 'Mahabaleshwar', 'Munnar', 'Ooty', 'Rameswaram'],
        'West': ['Gokarna']
    }
    filtered_data = data[data['Destination'].isin(direction_destinations.get(direction, data['Destination'].tolist()))]
    if filtered_data.empty:
        filtered_data = data
        filtered_scaled_features = scaled_features
    else:
        filtered_scaled_features = scaler.transform(encoder.transform(filtered_data[features]))
    rf_model.fit(filtered_scaled_features, filtered_data["Destination"])
    recommended_dest = rf_model.predict(input_scaled)[0].title() if option in ['destination', 'both'] else None
    session['recommended_dest'] = recommended_dest
    session['form_data'] = user_input.to_dict()
    session['starting_city'] = user_input.get('city')
    image_file = destination_images.get(recommended_dest, "default.jpg")
    image_url = f"/static/img/{image_file}" if os.path.exists(os.path.join(app.static_folder, 'img', image_file)) else "/static/img/default.jpg"
    if option == 'user':
        similarities = cosine_similarity(input_scaled, scaled_features)
        top_indices = np.argsort(similarities[0])[::-1][:4]
        matched_users = [data.iloc[i][["Name", "Email", "Phone Number", "Gender"]].to_dict() for i in top_indices]
        session['matched_users'] = matched_users
        session['current_match_index'] = 0
        return render_template('finalresult.html', matched_user=matched_users[0])
    return render_template('resultt.html', destination=recommended_dest, form_data=user_input, option=option, image_url=image_url)

@app.route('/match_users', methods=['POST'])
@token_required
def match_users():
    user_input = request.form
    input_df = pd.DataFrame([user_input])
    input_encoded = encoder.transform(input_df[features])
    input_scaled = scaler.transform(input_encoded)
    similarities = cosine_similarity(input_scaled, scaled_features)
    top_indices = np.argsort(similarities[0])[::-1][:4]
    matched_users = [data.iloc[i][["Name", "Email", "Phone Number", "Gender"]].to_dict() for i in top_indices]
    session['matched_users'] = matched_users
    session['current_match_index'] = 0
    return render_template('finalresult.html', matched_user=matched_users[0])

@app.route('/reject_user', methods=['POST'])
@token_required
def reject_user():
    matched_users = session.get('matched_users', [])
    current_index = session.get('current_match_index', 0)
    if current_index + 1 < len(matched_users) and current_index < 3:
        session['current_match_index'] = current_index + 1
        return render_template('finalresult.html', matched_user=matched_users[current_index + 1])
    return render_template('finalresult.html', matched_user=None, message="No more matches available")

@app.route('/accept_user', methods=['POST'])
@token_required
def accept_user():
    matched_users = session.get('matched_users', [])
    current_index = session.get('current_match_index', 0)
    starting_city = session.get('starting_city', 'your location')
    if matched_users and current_index < len(matched_users):
        accepted_user = matched_users[current_index]
        session['accepted_user'] = accepted_user
        message = f"You both are matched and can start from {starting_city}!"
        return render_template('finalresult.html', matched_user=accepted_user, show_message_form=True, message=message)
    return render_template('finalresult.html', matched_user=None, message="No user available")

@app.route('/send_message', methods=['POST'])
@token_required
def send_message():
    sender_username = jwt.decode(session['token'], 'mysecretkey123', algorithms=["HS256"])['user']
    sender = User.query.filter_by(username=sender_username).first()
    recipient_name = request.form.get('recipient_name')
    content = request.form.get('message')
    starting_city = session.get('starting_city', 'your location')
    if sender and recipient_name and content:
        new_message = Message(sender_id=sender.id, recipient_name=recipient_name, content=content)
        db.session.add(new_message)
        db.session.commit()
        return render_template('finalresult.html', matched_user=session.get('accepted_user'), message=f"Message sent! You both can start from {starting_city}!")
    return render_template('finalresult.html', matched_user=session.get('accepted_user'), message="Failed to send message")

@app.route('/notifications')
@token_required
def notifications():
    username = jwt.decode(session['token'], 'mysecretkey123', algorithms=["HS256"])['user']
    user = User.query.filter_by(username=username).first()
    sent_messages = Message.query.filter_by(sender_id=user.id).all()
    received_messages = Message.query.filter_by(recipient_name=username).all()
    return render_template('notifications.html', sent_messages=sent_messages, received_messages=received_messages)

@app.route('/plan_trip', methods=['GET', 'POST'])
@token_required
def plan_trip():
    if request.method == 'POST':
        destination = request.form.get('destination')
        if not destination:
            return render_template('resultt.html', message="No destination selected")

        session['trip_destination'] = destination
        starting_city = session.get('starting_city')
        preferences = session.get('form_data', {})
        
        dest_itinerary = itinerary_data[itinerary_data["Destination"].str.strip().str.lower() == destination.strip().lower()]
        max_days = int(dest_itinerary['Day'].max()) if not dest_itinerary.empty else 6
        
        days = min(int(request.form.get('days', max_days)), max_days) if request.form.get('days') else max_days
        
        itinerary, max_days_available = get_dynamic_itinerary(destination, days, preferences)
        if itinerary is None or not itinerary:
            return render_template('tripplan.html', 
                                 destination=destination,
                                 days=days,
                                 itinerary=[],
                                 custom_options={'hotels': [], 'restaurants': [], 'activities': []},
                                 travel_details=None,
                                 travel_costs=None,
                                 accommodation={'type': 'N/A', 'cost_per_night': 0, 'total_cost': 0, 'rating': 'N/A'},
                                 solo_total=0,
                                 buddy_total=0,
                                 daily_breakdown=[],
                                 budget=get_destination_budget(destination),
                                 activities_cost=0,
                                 error=f"No itinerary available for {destination}. Maximum days available: {max_days_available or 0}.")

        itinerary = [
            {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v 
             for k, v in day.items()}
            for day in itinerary
        ]

        travel_costs = get_travel_costs(starting_city, destination, days) if starting_city else None
        if travel_costs:
            selected_travel_option = 'train_sleeper' if travel_costs.get('train') else 'flight'
            selected_travel_cost = (travel_costs['train']['sleeper']['cost'] if selected_travel_option == 'train_sleeper' 
                                   else travel_costs['flight']['cost'])
            travel_time = (travel_costs['train']['sleeper']['time'] if selected_travel_option == 'train_sleeper' 
                          else travel_costs['flight']['time'])
            travel_details = {
                'mode': 'Train Sleeper' if selected_travel_option == 'train_sleeper' else 'Flight',
                'cost': selected_travel_cost,
                'time': travel_time,
                'departure': travel_costs['train']['sleeper']['departure'] if selected_travel_option == 'train_sleeper' else None,
                'arrival': travel_costs['train']['sleeper']['arrival'] if selected_travel_option == 'train_sleeper' else None,
                'maps_url': travel_costs['road']['maps_url'] if travel_costs.get('road') else None,
                'selected_option': selected_travel_option
            }
        else:
            selected_travel_cost = 0
            travel_details = None
        
        accommodation = get_accommodation(destination, days, 'mid-range')
        accommodation = {
            k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v 
            for k, v in accommodation.items()
        }

        activities_cost = int(sum(parse_cost(day["Daily Cost (₹)"]) for day in itinerary))
        solo_total = int(selected_travel_cost + accommodation['total_cost'] + activities_cost)
        buddy_total = int(round(selected_travel_cost * 0.75 + accommodation['total_cost'] * 0.5 + activities_cost * 0.8))
        
        daily_breakdown = []
        remaining_solo = solo_total
        remaining_buddy = buddy_total
        for day in itinerary:
            day_cost = int(parse_cost(day["Daily Cost (₹)"]))
            buddy_cost = int(day_cost * 0.8)
            remaining_solo -= day_cost
            remaining_buddy -= buddy_cost
            daily_breakdown.append({
                'day': int(day['Day']),
                'solo_spent': day_cost,
                'solo_remaining': int(max(0, remaining_solo)),
                'buddy_spent': buddy_cost,
                'buddy_remaining': int(max(0, remaining_buddy))
            })
        
        custom_options = {
            'hotels': [],
            'restaurants': [],
            'activities': []
        }
        if not dest_itinerary.empty:
            for _, row in dest_itinerary.iterrows():
                if pd.notna(row.get('Hotel', '')):
                    hotel_cost = int(parse_cost(row.get('Hotel Cost (₹)', 0)))
                    custom_options['hotels'].append((row['Hotel'], hotel_cost))
                
                if pd.notna(row.get('Restaurants', '')):
                    restaurants = row['Restaurants'].split(', ')
                    restaurant_cost = int(parse_cost(row.get('Restaurant Cost (₹)', 0)))
                    per_restaurant_cost = int(restaurant_cost // len(restaurants)) if restaurants else 0
                    for restaurant in restaurants:
                        custom_options['restaurants'].append((restaurant, per_restaurant_cost))
                
                if pd.notna(row.get('Activities', '')):
                    activities = row['Activities'].split(', ')
                    activity_cost = int(parse_cost(row.get('Activity Cost (₹)', 0)))
                    per_activity_cost = int(activity_cost // len(activities)) if activities else 0
                    for activity in activities:
                        custom_options['activities'].append((activity, per_activity_cost))

        if not any(custom_options.values()):
            for day in itinerary:
                for time in ["Morning Activity (Cost)", "Afternoon Activity (Cost)", "Evening Activity (Cost)"]:
                    activity = day[time]
                    cost = extract_activity_costs(activity)
                    if "Hotel" in activity or "Stay" in activity:
                        custom_options['hotels'].append((activity, cost))
                    elif "Cafe" in activity or "Dinner" in activity or "Restaurant" in activity:
                        custom_options['restaurants'].append((activity, cost))
                    else:
                        custom_options['activities'].append((activity, cost))

        custom_options['hotels'] = list(set(custom_options['hotels']))
        custom_options['restaurants'] = list(set(custom_options['restaurants']))
        custom_options['activities'] = list(set(custom_options['activities']))

        session['original_itinerary'] = itinerary
        session['custom_options'] = custom_options

        budget = get_destination_budget(destination)

        return render_template('tripplan.html',
                              itinerary=itinerary,
                              destination=destination,
                              starting_city=starting_city,
                              travel_details=travel_details,
                              travel_costs=travel_costs,
                              accommodation=accommodation,
                              solo_total=solo_total,
                              buddy_total=buddy_total,
                              daily_breakdown=daily_breakdown,
                              days=days,
                              max_days=max_days,
                              custom_options=custom_options,
                              budget=budget,
                              activities_cost=activities_cost)
    return redirect('/home')

@app.route('/save_custom_plan', methods=['POST'])
@token_required
def save_custom_plan():
    destination = session.get('trip_destination')
    starting_city = session.get('starting_city')
    days = int(request.form.get('days'))
    
    dest_itinerary = itinerary_data[itinerary_data["Destination"].str.strip().str.lower() == destination.strip().lower()]
    max_days = int(dest_itinerary['Day'].max()) if not dest_itinerary.empty else 6
    
    if days > max_days:
        return render_template('tripplan.html', 
                              itinerary=session.get('original_itinerary', []),
                              destination=destination,
                              starting_city=starting_city,
                              travel_details=None,
                              travel_costs=None,
                              accommodation={'type': 'N/A', 'cost_per_night': 0, 'total_cost': 0, 'rating': 'N/A'},
                              solo_total=0,
                              buddy_total=0,
                              daily_breakdown=[],
                              days=days,
                              max_days=max_days,
                              custom_options=session.get('custom_options', {'hotels': [], 'restaurants': [], 'activities': []}),
                              budget=get_destination_budget(destination),
                              activities_cost=0,
                              error=f"Days cannot exceed maximum available: {max_days}")

    travel_costs = get_travel_costs(starting_city, destination, days) if starting_city else None
    selected_travel_option = request.form.get('travel_option', 'train_sleeper' if travel_costs.get('train') else 'flight')
    
    if travel_costs:
        if selected_travel_option.startswith('train_') and travel_costs.get('train'):
            tier = selected_travel_option.split('_')[1]
            selected_travel_cost = travel_costs['train'][tier]['cost']
            travel_time = travel_costs['train'][tier]['time']
            travel_details = {
                'mode': f"Train {tier.capitalize()}",
                'cost': selected_travel_cost,
                'time': travel_time,
                'departure': travel_costs['train'][tier]['departure'],
                'arrival': travel_costs['train'][tier]['arrival'],
                'maps_url': None,
                'selected_option': selected_travel_option
            }
        elif selected_travel_option == 'flight':
            selected_travel_cost = travel_costs['flight']['cost']
            travel_time = travel_costs['flight']['time']
            travel_details = {
                'mode': 'Flight',
                'cost': selected_travel_cost,
                'time': travel_time,
                'departure': None,
                'arrival': None,
                'maps_url': None,
                'selected_option': selected_travel_option
            }
        elif selected_travel_option == 'road' and travel_costs.get('road'):
            selected_travel_cost = travel_costs['road']['cost']
            travel_time = travel_costs['road']['time']
            travel_details = {
                'mode': 'Road',
                'cost': selected_travel_cost,
                'time': travel_time,
                'departure': None,
                'arrival': None,
                'maps_url': travel_costs['road']['maps_url'],
                'selected_option': selected_travel_option
            }
        else:
            selected_travel_cost = 0
            travel_details = None
    else:
        selected_travel_cost = 0
        travel_details = None

    custom_plan = []
    for i in range(days):
        morning = request.form.get(f'day_{i}_morning', 'Free Time (₹0)')
        afternoon = request.form.get(f'day_{i}_afternoon', 'Free Time (₹0)')
        evening = request.form.get(f'day_{i}_evening', 'Free Time (₹0)')
        day_data = {
            'Day': i + 1,
            'Morning Activity (Cost)': morning,
            'Afternoon Activity (Cost)': afternoon,
            'Evening Activity (Cost)': evening,
            'Daily Cost (₹)': 0
        }
        daily_cost = (extract_activity_costs(morning) +
                      extract_activity_costs(afternoon) +
                      extract_activity_costs(evening))
        day_data['Daily Cost (₹)'] = daily_cost
        custom_plan.append(day_data)

    accommodation = get_accommodation(destination, days, 'mid-range')
    activities_cost = int(sum(day['Daily Cost (₹)'] for day in custom_plan))
    solo_total = int(selected_travel_cost + accommodation['total_cost'] + activities_cost)
    buddy_total = int(round(selected_travel_cost * 0.75 + accommodation['total_cost'] * 0.5 + activities_cost * 0.8))

    daily_breakdown = []
    remaining_solo = solo_total
    remaining_buddy = buddy_total
    for day in custom_plan:
        day_cost = int(day['Daily Cost (₹)'])
        buddy_cost = int(day_cost * 0.8)
        remaining_solo -= day_cost
        remaining_buddy -= buddy_cost
        daily_breakdown.append({
            'day': day['Day'],
            'solo_spent': day_cost,
            'solo_remaining': int(max(0, remaining_solo)),
            'buddy_spent': buddy_cost,
            'buddy_remaining': int(max(0, remaining_buddy))
        })

    custom_options = session.get('custom_options', {'hotels': [], 'restaurants': [], 'activities': []})
    budget = get_destination_budget(destination)

    return render_template('tripplan.html',
                          itinerary=custom_plan,
                          destination=destination,
                          starting_city=starting_city,
                          travel_details=travel_details,
                          travel_costs=travel_costs,
                          accommodation=accommodation,
                          solo_total=solo_total,
                          buddy_total=buddy_total,
                          daily_breakdown=daily_breakdown,
                          days=days,
                          max_days=max_days,
                          custom_options=custom_options,
                          budget=budget,
                          activities_cost=activities_cost)

@app.route('/expenses', methods=['GET', 'POST'])
@token_required
def expenses():
    if request.method == 'POST':
        amount = float(request.form.get('amount'))
        category = request.form.get('category')
        description = request.form.get('description', '')
        username = jwt.decode(session['token'], 'mysecretkey123', algorithms=["HS256"])['user']
        user = User.query.filter_by(username=username).first()
        new_expense = Expense(user_id=user.id, amount=amount, category=category, description=description)
        db.session.add(new_expense)
        db.session.commit()
        return redirect('/expenses')
    username = jwt.decode(session['token'], 'mysecretkey123', algorithms=["HS256"])['user']
    user = User.query.filter_by(username=username).first()
    user_expenses = Expense.query.filter_by(user_id=user.id).all()
    return render_template('expenses.html', expenses=user_expenses)

@app.route('/dashboard')
@token_required
def dashboard():
    username = jwt.decode(session['token'], 'mysecretkey123', algorithms=["HS256"])['user']
    user = User.query.filter_by(username=username).first()
    user_expenses = Expense.query.filter_by(user_id=user.id).all()
    recommended_dest = session.get('recommended_dest')
    form_data = session.get('form_data')
    option = session.get('option')
    matched_users = session.get('matched_users', [])
    current_index = session.get('current_match_index', 0)
    matched_user = matched_users[current_index] if matched_users and current_index < len(matched_users) else None
    destination = session.get('trip_destination')
    itinerary = itinerary_data[itinerary_data["Destination"] == destination].to_dict(orient='records') if destination else []
    return render_template('dashboard.html', expenses=user_expenses, recommended_dest=recommended_dest,
                          form_data=form_data, option=option, matched_user=matched_user,
                          itinerary=itinerary, destination=destination)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)