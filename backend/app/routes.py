# app/routes.py
from flask import Blueprint, jsonify, request
from app.ml_model import WildfirePredictor
from app.gemini_chatbot import FirePreventionChatbot

bp = Blueprint('routes', __name__)

# Initialize and train model once when server starts
print("Initializing wildfire prediction model...")
predictor = WildfirePredictor(csv_path="./data/viirs-jpss1_2024_Canada.csv")
predictor.train()

# Initialize Gemini chatbot
print("Initializing Gemini fire prevention chatbot...")
try:
    chatbot = FirePreventionChatbot()
    print("Gemini chatbot initialized successfully!")
except Exception as e:
    print(f"Warning: Gemini chatbot initialization failed: {e}")
    print("Make sure to set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
    chatbot = None
    
# Dummy data for locations without ML predictions or if predictions fail
WILDFIRE_DATA = {
    'british columbia': [
        {'latitude': 49.2827, 'longitude': -123.1207, 'name': 'Vancouver Fire', 'severity': 'moderate'},
        {'latitude': 50.6745, 'longitude': -120.3273, 'name': 'Kamloops Fire', 'severity': 'high'},
    ]
}

# Location name to coordinates mapping
LOCATION_COORDS = {
    'bc': (54.0, -125.0),  # Center of BC
    'vancouver': (49.2827, -123.1207),
    'kamloops': (50.6745, -120.3273),
    'prince george': (53.9171, -122.7497),
    'surrey' : (49.1913, -122.8490),
    'richmond' : (49.1666, -123.1336),
    'burnaby' : (49.2488, -122.9805),
    'victoria' : (48.4284, -123.3656),
    'quesnel' : (52.9840, -122.4930),
    'kelowna' : (49.8880, -119.4960),
}

@bp.route('/api/search', methods=['POST'])
def search_fires():
    """
    Search for fires by location name using ML predictions
    """
    data = request.get_json()
    location = data.get('q', '').lower().strip()
    
    if not location:
        return jsonify({'error': 'Location query required'}), 400
    
    # Try to get coordinates for the location
    coords = None
    for loc_name, loc_coords in LOCATION_COORDS.items():
        if location in loc_name or loc_name in location:
            coords = loc_coords
            break
    
    if not coords:
        return jsonify({
            'fires': [],
            'message': f'Location "{location}" not recognized. Try: British Columbia, Vancouver, Kamloops'
        }), 404
    
    try:
        # Get ML predictions
        predicted_fires = predictor.predict(
            center_lat=coords[0],
            center_lon=coords[1],
            search_radius_km=100,
            top_n=10
        )

        # Ensure each predicted fire has 'severity' and 'confidence' fields
        def ensure_fields(fire):
            f = fire.copy()
            prob = float(f.get('probability') or f.get('prob') or 0.0)
            if 'severity' not in f or not f.get('severity'):
                f['severity'] = 'high' if prob >= 0.75 else ('moderate' if prob >= 0.4 else ('low' if prob > 0 else 'unknown'))
            if 'confidence' not in f:
                f['confidence'] = prob
            # Normalize key names: frontend expects 'latitude'/'longitude'
            if 'lat' in f and 'latitude' not in f:
                f['latitude'] = float(f['lat'])
            if 'lon' in f and 'longitude' not in f:
                f['longitude'] = float(f['lon'])
            return f

        augmented = [ensure_fields(f) for f in predicted_fires]

        print(f"[DEBUG] Returning {len(augmented)} predicted fires for '{location}' (center={coords})")
        if augmented:
            print("[DEBUG] sample fire:", augmented[0])

        return jsonify({
            'fires': augmented,
            'count': len(augmented),
            'location': location,
            'center': {'lat': coords[0], 'lon': coords[1]},
            'source': 'ML Prediction'
        })
        
    except Exception as e:
        print(f"ML prediction error: {e}")
        # Fallback to dummy data
        dummy_fires = WILDFIRE_DATA.get(location, [])
        return jsonify({
            'fires': dummy_fires,
            'count': len(dummy_fires),
            'location': location,
            'source': 'Fallback Data'
        })

@bp.route('/api/predict-fires', methods=['POST'])
def predict_fires_by_coords():
    """
    Predict fires by exact coordinates (lat/lon)
    """
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    radius = data.get('radius', 50)
    
    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon required'}), 400
    
    try:
        predicted_fires = predictor.predict(
            center_lat=float(lat),
            center_lon=float(lon),
            search_radius_km=radius,
            top_n=10
        )
        
        return jsonify({
            'fires': predicted_fires,
            'count': len(predicted_fires),
            'center': {'lat': lat, 'lon': lon}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/chatbot/welcome', methods=['GET'])
def chatbot_welcome():
    """
    Get welcome message from the fire prevention chatbot
    """
    if not chatbot:
        return jsonify({
            'success': False,
            'error': 'Chatbot not available. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.'
        }), 503
    
    try:
        response = chatbot.get_welcome_message()
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get welcome message: {str(e)}'
        }), 500

@bp.route('/api/chatbot/chat', methods=['POST'])
def chatbot_chat():
    """
    Send a message to the fire prevention chatbot
    """
    if not chatbot:
        return jsonify({
            'success': False,
            'error': 'Chatbot not available. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.'
        }), 503
    
    data = request.get_json()
    message = data.get('message', '').strip()
    conversation_history = data.get('conversation_history', [])
    
    if not message:
        return jsonify({
            'success': False,
            'error': 'Message is required'
        }), 400
    
    try:
        response = chatbot.chat(message, conversation_history)
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Chatbot error: {str(e)}'
        }), 500