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
import requests
import time

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

# Load datasets (unchanged)
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

# Helper Functions (unchanged)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def get_dynamic_itinerary(destination, days, preferences):
    dest_itinerary = itinerary_data[itinerary_data["Destination"].str.strip().str.lower() == destination.strip().lower()]
    if not dest_itinerary.empty:
        itinerary = dest_itinerary.head(days).to_dict(orient='records')
        if len(itinerary) < days:
            repeated = dest_itinerary.to_dict(orient='records')
            while len(itinerary) < days:
                for entry in repeated:
                    if len(itinerary) < days:
                        itinerary.append(entry)
    else:
        url = f"https://nominatim.openstreetmap.org/search?q={destination}&format=json&limit=1"
        headers = {'User-Agent': 'TravelPlannerApp/1.0'}
        response = requests.get(url, headers=headers).json()
        if not response:
            return [{'Day': i + 1, 'Morning Activity (Cost)': f"Explore {destination} (₹500)", 
                     'Afternoon Activity (Cost)': f"Sightseeing in {destination} (₹700)", 
                     'Evening Activity (Cost)': f"Relax in {destination} (₹600)", 'Daily Cost (₹)': 1800} for i in range(days)]
        
        lat, lon = float(response[0]['lat']), float(response[0]['lon'])
        time.sleep(1)
        
        poi_types = ['tourism', 'historic', 'natural', 'leisure']
        activities = []
        for poi_type in poi_types:
            time.sleep(1)
            search_url = f"https://nominatim.openstreetmap.org/search?q={poi_type}+near+{destination}&format=json&limit=5"
            pois = requests.get(search_url, headers=headers).json()
            activities.extend([poi.get('display_name', f"{poi_type.capitalize()} Spot") for poi in pois])
        
        if not activities:
            activities = [f"Explore {destination} Area"] * 3 * days
        
        itinerary = []
        for i in range(days):
            day_activities = {
                'Day': i + 1,
                'Morning Activity (Cost)': f"{activities[i % len(activities)]} (₹500)",
                'Afternoon Activity (Cost)': f"{activities[(i+1) % len(activities)]} (₹700)",
                'Evening Activity (Cost)': f"{activities[(i+2) % len(activities)]} (₹600)",
                'Daily Cost (₹)': 1800
            }
            itinerary.append(day_activities)

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
                        cost = int(cost_str.strip(')').replace(',', '')) + 200
                        adjusted_day[time_slot] = f"Adventure {activity} (₹{cost})"
                    except (ValueError, IndexError):
                        adjusted_day[time_slot] = f"Adventure {adjusted_day[time_slot].split(' (₹')[0]} (₹500)"
        elif 'relaxing' in vibe:
            for time_slot in ['Morning Activity (Cost)', 'Afternoon Activity (Cost)', 'Evening Activity (Cost)']:
                if time_slot in adjusted_day and adjusted_day[time_slot]:
                    try:
                        activity, cost_str = adjusted_day[time_slot].rsplit(' (₹', 1)
                        cost = int(cost_str.strip(')').replace(',', '')) - 100
                        adjusted_day[time_slot] = f"Relax at {activity} (₹{cost})"
                    except (ValueError, IndexError):
                        adjusted_day[time_slot] = f"Relax at {adjusted_day[time_slot].split(' (₹')[0]} (₹400)"
        if 'Daily Cost (₹)' in adjusted_day:
            daily_cost = 0
            for time_slot in ['Morning Activity (Cost)', 'Afternoon Activity (Cost)', 'Evening Activity (Cost)']:
                if time_slot in adjusted_day and adjusted_day[time_slot]:
                    try:
                        cost_str = adjusted_day[time_slot].rsplit('₹', 1)[1].strip(')')
                        cost = int(cost_str.replace(',', ''))
                        daily_cost += cost
                    except (ValueError, IndexError):
                        daily_cost += 500
            adjusted_day['Daily Cost (₹)'] = daily_cost
        adjusted_itinerary.append(adjusted_day)
    return adjusted_itinerary

def get_travel_costs(starting_city, destination, days):
    try:
        start_lat = city_coords.loc[starting_city, 'Latitude']
        start_lon = city_coords.loc[starting_city, 'Longitude']
        dest_lat = city_coords.loc[destination, 'Latitude']
        dest_lon = city_coords.loc[destination, 'Longitude']
        distance = haversine_distance(start_lat, start_lon, dest_lat, dest_lon)
        road_distance = distance * 1.2
        
        flight_cost = 2500 + (distance * 6)
        train_costs = {
            'sleeper': round(distance * 0.8 + 400),
            'ac3': round(distance * 1.6 + 800),
            'ac2': round(distance * 2.4 + 1200)
        }
        days_road = max(1, round(road_distance / 600))
        road_cost = round((road_distance * 12) + 1500 + (500 * days_road))
        
        is_island = destination in ["Andaman", "Lakshadweep"]
        return {
            'distance': round(distance),
            'road_distance': round(road_distance),
            'flight': round(flight_cost * (1.5 if is_island else 1)),
            'train': None if is_island else train_costs,
            'road': None if is_island else road_cost,
            'days': days_road
        }
    except KeyError as e:
        print(f"Coordinates not found for {e}")
        return None

def get_accommodation(destination, days, budget_level='mid-range'):
    dest_info = data[data['Destination'].str.lower() == destination.lower()].iloc[0] if destination.lower() in data['Destination'].str.lower().values else None
    base_budget = int(dest_info['Budget (₹)'].replace(',', '')) if dest_info is not None and 'Budget (₹)' in dest_info else 5000
    options = {
        'budget': {'cost_per_night': base_budget // 5, 'type': 'Hostel', 'rating': '3.5/5'},
        'mid-range': {'cost_per_night': base_budget // 2, 'type': 'Hotel', 'rating': '4.0/5'},
        'luxury': {'cost_per_night': base_budget, 'type': 'Resort', 'rating': '4.8/5'}
    }
    choice = options.get(budget_level, options['mid-range'])
    return {'type': choice['type'], 'total_cost': choice['cost_per_night'] * days, 'rating': choice['rating']}

def get_local_logistics(destination):
    dest_info = data[data['Destination'].str.lower() == destination.lower()].iloc[0] if destination.lower() in data['Destination'].str.lower().values else None
    transport_cost = 400
    if dest_info and 'Hill' in dest_info['Type']:
        transport_cost += 150
    elif destination in ["Andaman", "Lakshadweep"]:
        transport_cost += 300
    return {
        'local_transport': {'type': 'Cab/Public', 'daily_cost': transport_cost},
        'visa': 'Required' if destination in ['Andaman', 'Lakshadweep'] else 'Not Required'
    }

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get('token')
        if not token:
            return redirect('/login?next=' + request.path)
        try:
            jwt.decode(token, 'mysecretkey123', algorithms=["HS256"])
        except:
            return redirect('/login?next=' + request.path)
        return f(*args, **kwargs)
    return decorated

# Routes (all unchanged except /plan_trip)
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

# Updated /plan_trip route with all 6 steps
@app.route('/plan_trip', methods=['GET', 'POST'])
@token_required
def plan_trip():
    if request.method == 'POST':
        destination = request.form.get('destination')
        if not destination:
            return render_template('resultt.html', message="No destination selected")

        # Step 3: Take number of days input
        days = int(request.form.get('days', 3))
        travel_mode = request.form.get('travel_mode', 'flight')
        budget_level = request.form.get('budget_level', 'mid-range')
        
        session['trip_destination'] = destination
        starting_city = session.get('starting_city')
        preferences = session.get('form_data', {})
        
        # Step 3: Adjust itinerary dynamically
        itinerary = get_dynamic_itinerary(destination, days, preferences)
        
        # Step 1 & 2: Plan travel and show generic costs
        travel_costs = get_travel_costs(starting_city, destination, days) if starting_city else None
        if travel_costs and travel_mode == 'train':
            selected_travel_cost = travel_costs['train']['sleeper'] if travel_costs['train'] else 0
        else:
            selected_travel_cost = travel_costs[travel_mode] if travel_costs and travel_mode in travel_costs else 0
        
        # Step 4 & 5: Calculate solo and buddy costs, adjust to budget
        budget_map = {'budget': 10000, 'mid-range': 25000, 'luxury': 50000}
        budget = budget_map.get(budget_level, 25000)
        
        solo_total = sum(int(day["Daily Cost (₹)"]) for day in itinerary) + selected_travel_cost
        buddy_total = sum(
            int(day["Daily Cost (₹)"]) * 0.5 if "hotel" in day.get("Morning Activity (Cost)", "").lower() or "taxi" in day.get("Evening Activity (Cost)", "").lower()
            else int(day["Daily Cost (₹)"])
            for day in itinerary
        ) + (selected_travel_cost * 0.75)  # 25% travel discount for buddy
        
        if solo_total > budget:
            reduction_factor = budget / solo_total
            for day in itinerary:
                original_cost = int(day["Daily Cost (₹)"])
                new_cost = int(original_cost * reduction_factor)
                day["Daily Cost (₹)"] = new_cost
                if original_cost > new_cost and day.get("Evening Activity (Cost)") and "free" not in day["Evening Activity (Cost)"].lower():
                    day["Evening Activity (Cost)"] = "Skipped to fit budget"
            solo_total = budget
            buddy_total = int(buddy_total * reduction_factor)

        # Step 6: Daily breakdown
        daily_breakdown = []
        remaining_solo = solo_total
        remaining_buddy = buddy_total
        for day in itinerary:
            day_cost = int(day["Daily Cost (₹)"])
            buddy_cost = day_cost * 0.5 if "hotel" in day.get("Morning Activity (Cost)", "").lower() else day_cost
            remaining_solo -= day_cost
            remaining_buddy -= buddy_cost
            daily_breakdown.append({
                'day': day['Day'],
                'solo_spent': day_cost,
                'buddy_spent': buddy_cost,
                'solo_remaining': max(0, remaining_solo),
                'buddy_remaining': max(0, remaining_buddy)
            })

        if travel_costs is None and starting_city:
            return render_template('resultt.html', message=f"Could not calculate travel costs for {starting_city} to {destination}")
        
        return render_template('tripplan.html', 
                              itinerary=itinerary, 
                              destination=destination, 
                              starting_city=starting_city, 
                              travel_costs=travel_costs, 
                              travel_mode=travel_mode,
                              solo_total=solo_total, 
                              buddy_total=buddy_total, 
                              budget=budget,
                              daily_breakdown=daily_breakdown,
                              days=days)
    else:
        recommended_dest = session.get('recommended_dest')
        if recommended_dest:
            return render_template('resultt.html', destination=recommended_dest, 
                                  message="Please use the form to plan your trip.")
        return redirect('/home')

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