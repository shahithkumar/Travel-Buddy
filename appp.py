import requests

# API key
RAPIDAPI_KEY = "1edd748728msheade9a5f4b09191p1084d5jsn4cf086ecd58d"

# API hosts
RAPIDAPI_HOST_IRCTC = "irctc1.p.rapidapi.com"
RAPIDAPI_HOST_TRIPADVISOR = "tripadvisor-scraper.p.rapidapi.com"
RAPIDAPI_HOST_TRIPADVISOR_V2 = "real-time-tripadvisor-scraper-api.p.rapidapi.com"
RAPIDAPI_HOST_GOOGLE_MAPS = "google-maps-api-free.p.rapidapi.com"
RAPIDAPI_HOST_DRIVING_DISTANCE = "driving-distance-calculator-between-two-points.p.rapidapi.com"
RAPIDAPI_HOST_IMAGE_SEARCH = "real-time-image-search.p.rapidapi.com"
RAPIDAPI_HOST_SKYSCANNER = "sky-scanner3.p.rapidapi.com"

# Mapping of city names to station codes
city_to_station_code = {
    "Puri": "PURI",
    "Guwahati": "GHY",
}

# Test 1: IRCTC API - Train between Puri and Guwahati
def test_irctc_api():
    url = f"https://{RAPIDAPI_HOST_IRCTC}/api/v3/trainBetweenStations"
    querystring = {
        "fromStationCode": city_to_station_code["Puri"],
        "toStationCode": city_to_station_code["Guwahati"],
        "dateOfJourney": "2025-04-01"
    }
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_IRCTC,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting IRCTC API: Train between Puri and Guwahati...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=20)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        if data.get("status") and data.get("data"):
            train = data["data"][0]
            print("Sample Response (First Train):")
            print(f"Train Name: {train.get('train_name')}")
            print(f"Train Number: {train.get('train_number')}")
            print(f"Distance: {train.get('distance')} km")
            print("Status: Working")
        else:
            print("No train data found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching train data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 2: TripAdvisor Scraper API - Hotels in Cherrapunji
def test_tripadvisor_hotels():
    url = f"https://{RAPIDAPI_HOST_TRIPADVISOR}/hotels/list"
    querystring = {"query": "Cherrapunji", "page": "1"}
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_TRIPADVISOR,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting TripAdvisor Scraper API: Hotels in Cherrapunji...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        if data.get("results") and isinstance(data["results"], list) and data["results"]:
            hotel = data["results"][0]
            print("Sample Response (First Hotel):")
            print(f"Hotel Name: {hotel.get('name')}")
            print(f"Location: {hotel.get('address')}")
            print(f"Rating: {hotel.get('rating')}")
            print("Status: Working")
        else:
            print("No hotels found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching hotels: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 3: TripAdvisor Scraper API - Restaurants in Cherrapunji
def test_tripadvisor_restaurants():
    url = f"https://{RAPIDAPI_HOST_TRIPADVISOR}/restaurants/list"
    querystring = {"query": "Sohra", "page": "1"}
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_TRIPADVISOR,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting TripAdvisor Scraper API: Restaurants in Cherrapunji (Sohra)...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        if data.get("results") and isinstance(data["results"], list) and data["results"]:
            restaurant = data["results"][0]
            print("Sample Response (First Restaurant):")
            print(f"Restaurant Name: {restaurant.get('name')}")
            print(f"Address: {restaurant.get('address')}")
            print(f"Rating: {restaurant.get('rating')}")
            print("Status: Working")
        else:
            print("No restaurants found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching restaurants: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 4: Real-Time TripAdvisor Scraper API - Restaurants in New York
def test_tripadvisor_v2_restaurants():
    url = f"https://{RAPIDAPI_HOST_TRIPADVISOR_V2}/tripadvisor_restaurants_search_v2"
    querystring = {"location": "new york"}
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_TRIPADVISOR_V2,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting Real-Time TripAdvisor Scraper API: Restaurants in New York...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        if data.get("data") and isinstance(data["data"], list) and data["data"]:
            restaurant = data["data"][0]
            print("Sample Response (First Restaurant):")
            print(f"Restaurant Name: {restaurant.get('name')}")
            print(f"Address: {restaurant.get('address')}")
            print(f"Rating: {restaurant.get('rating')}")
            print("Status: Working")
        else:
            print("No restaurants found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching restaurants: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 5: SkyScanner API - Flights Search One-Way from Paris
def test_skyscanner_flights_search():
    url = f"https://{RAPIDAPI_HOST_SKYSCANNER}/flights/search-one-way"
    querystring = {
        "fromEntityId": "PARI",  # Paris
        "toEntityId": "NYCA",  # New York (added for specificity)
        "departDate": "2025-03-25",  # Added a closer date
        "cabinClass": "economy",
        "market": "US",
        "locale": "en-US",
        "currency": "USD",
        "adults": "1"
    }
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_SKYSCANNER,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting SkyScanner API: Flights Search One-Way from Paris to New York...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=20)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        # Check if the status is "incomplete" and requires using the search-incomplete endpoint
        if data.get("context", {}).get("status") == "incomplete":
            print("Status is 'incomplete'. Need to use /flights/search-incomplete endpoint...")
            # Simulate a follow-up request (in a real scenario, you'd need to poll the search-incomplete endpoint)
            print("Assuming follow-up request completes...")
        if data.get("itineraries", {}).get("results"):
            flight = data["itineraries"]["results"][0]
            print("Sample Response (First Flight):")
            print(f"Flight ID: {flight.get('id')}")
            print(f"Price: {flight.get('price', {}).get('formatted')}")
            print(f"Departure: {flight.get('legs', [{}])[0].get('departure')}")
            print(f"Arrival: {flight.get('legs', [{}])[0].get('arrival')}")
            print("Status: Working")
        else:
            print("No flights found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching flight data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 6: Google Maps API - Nearby Restaurants (Mumbai area)
def test_google_maps_api():
    url = f"https://{RAPIDAPI_HOST_GOOGLE_MAPS}/google-nearby-search"
    querystring = {
        "lat": "19.24232736426361",
        "long": "72.85841985686734",
        "radius": "1000",
        "type": "restaurant",
        "keyword": "local"
    }
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_GOOGLE_MAPS,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting Google Maps API: Nearby Restaurants (Mumbai area)...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        if data.get("results"):
            restaurant = data["results"][0]
            print("Sample Response (First Restaurant):")
            print(f"Restaurant Name: {restaurant.get('name')}")
            print(f"Address: {restaurant.get('vicinity')}")
            print(f"Rating: {restaurant.get('rating')}")
            print("Status: Working")
        else:
            print("No restaurants found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching nearby restaurants: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 7: Driving Distance Calculator API - Distance from New York City to Jersey City
def test_driving_distance_api():
    url = f"https://{RAPIDAPI_HOST_DRIVING_DISTANCE}/data"
    querystring = {
        "origin": "New York City, NY",
        "destination": "Jersey City, Hudson County"
    }
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_DRIVING_DISTANCE,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting Driving Distance API: Distance from New York City to Jersey City...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print("Sample Response:")
        print(f"Distance: {data.get('distance_in_kilometers')} km")
        print(f"Duration: {data.get('travel_time')}")
        print("Status: Working")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching road distance: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Test 8: Real-Time Image Search API - Images for "beach"
def test_image_search_api():
    url = f"https://{RAPIDAPI_HOST_IMAGE_SEARCH}/search"
    querystring = {
        "query": "beach",
        "limit": "10",
        "size": "any",
        "color": "any",
        "type": "any",
        "time": "any",
        "usage_rights": "any",
        "file_type": "any",
        "aspect_ratio": "any",
        "safe_search": "off",
        "region": "us"
    }
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST_IMAGE_SEARCH,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    try:
        print("\nTesting Real-Time Image Search API: Images for 'beach'...")
        print("Input Query Parameters:", querystring)
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        data = response.json()
        if data.get("data"):
            image = data["data"][0]
            print("Sample Response (First Image):")
            print(f"Image URL: {image.get('url')}")
            print(f"Title: {image.get('title')}")
            print("Status: Working")
        else:
            print("No images found in response.")
            print("Status: Not Working (No Data)")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching images: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        print("Status: Not Working (Error)")

# Run all tests
if __name__ == "__main__":
    print("Starting API Tests...\n")
    test_irctc_api()
    test_tripadvisor_hotels()
    test_tripadvisor_restaurants()
    test_tripadvisor_v2_restaurants()
    test_skyscanner_flights_search()
    test_google_maps_api()
    test_driving_distance_api()
    test_image_search_api()
    print("\nAll API Tests Completed.")