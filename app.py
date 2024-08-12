from flask import Flask, render_template, request, jsonify
from models.draft_recommendation import FantasyFootballDraftAssistant
import logging
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize and load the draft assistant
draft_assistant = FantasyFootballDraftAssistant()
draft_assistant.load_data('data/cbs_fantasy_projection_master.csv')

model_file = 'models/fantasy_football_model.pkl'
if os.path.exists(model_file):
    draft_assistant.load_model(model_file)
    logging.info("Loaded existing model.")
else:
    logging.error("Pre-trained model not found. Please ensure the model file exists.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/draft')
def draft():
    return render_template('draft.html')

@app.route('/research')
def research():
    return render_template('research.html')

@app.route('/api/available_players', methods=['GET'])
def get_available_players():
    logging.debug("Received request for available players")
    players = draft_assistant.df.replace({np.nan: None}).to_dict('records')
    
    # Convert any remaining non-JSON-serializable values to strings
    for player in players:
        for key, value in player.items():
            if isinstance(value, (np.integer, np.floating)):
                player[key] = float(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                player[key] = str(value)
    
    logging.debug(f"Returning {len(players)} players")
    return jsonify(players)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    logging.debug("Received request for recommendations")
    data = request.json
    logging.debug(f"Recommendation request data: {data}")
    team = data.get('team', [])
    available_players = data.get('available_players', [])
    round_num = data.get('round_num', 1)
    pick_number = data.get('pick_number', 1)
    
    if not available_players:
        logging.error("No available players provided")
        return jsonify({"error": "No available players", "details": data}), 400
    
    try:
        # Convert team and available_players from list of names to list of player dictionaries
        team_dicts = [next(p for p in draft_assistant.players if p['player'] == player) for player in team]
        available_player_dicts = [next(p for p in draft_assistant.players if p['player'] == player) for player in available_players]
        
        recommendations = draft_assistant.recommend_players(team_dicts, available_player_dicts, round_num, pick_number)
        
        if not recommendations:
            logging.error("No recommendations generated")
            return jsonify({"error": "No recommendations generated", "details": data}), 400
        
        logging.debug(f"Returning {len(recommendations)} recommendations")
        return jsonify(recommendations)
    except Exception as e:
        logging.exception("Unexpected error in get_recommendations")
        return jsonify({"error": str(e), "details": data}), 500


if __name__ == '__main__':
    app.run(debug=True)