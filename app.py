from flask import Flask, render_template, request, jsonify
from models.draft_model import FantasyFootballDraftAssistant
import logging
import json
import numpy as np
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)

draft_assistant = FantasyFootballDraftAssistant()
draft_assistant.load_data('data//cbs_fantasy_projection_master.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/draft')
def draft():
    return render_template('draft.html')
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
    
    if not available_players:
        logging.error("No available players provided")
        return jsonify({"error": "No available players", "details": data}), 400
    
    recommendations = draft_assistant.recommend_players(team, available_players, round_num)
    
    if isinstance(recommendations, list) and recommendations and 'error' in recommendations[0]:
        logging.error(f"Error in recommendations: {recommendations[0]}")
        return jsonify(recommendations[0]), 400
    
    logging.debug(f"Returning {len(recommendations)} recommendations")
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)