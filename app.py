from flask import Flask, render_template, request, jsonify, Response
from models.draft_recommendation import FantasyFootballDraftAssistant
import logging
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import TiDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import json
import yaml
from NFLFantasyQA import NFLFantasyQA
from utils import create_retriever, create_chatbot, create_youtube_retriever, split_response,create_nfl_fantasy_chatbot
logger = logging.getLogger(__name__)


# Load config
with open('static/config.yml', 'r') as file:
    config = yaml.safe_load(file)

YOUTUBE_TABLE_NAME = config['vectordb']['youtube']
ARTICLE_TABLE_NAME = config['vectordb']['article']
PLAYER_REPORT_TABLE_NAME=config['vectordb']['playerreport']
EMBEDDING_MODEL = config['EMBEDDING_MODEL']


# Load environment variables
load_dotenv()
tidb_connection_string = os.getenv('TIDB_CONNECTION_URL')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize the NFLFantasyQA instance
nfl_fantasy_qa = NFLFantasyQA()
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)


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

# Constants for chatbot



# Routes
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

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    def generate():
        try:
            response = nfl_fantasy_qa.get_answer(question)
            
            content, references = split_response(response)
            
            for chunk in content.split('\n'):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            if references:
                yield f"data: {json.dumps({'references': references})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            logging.exception("Error in chatbot response generation")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), content_type='text/event-stream')

# Helper function to split the response into content and references
# def split_response(response):
#     parts = response.split("## References")
#     content = parts[0].strip()
#     references = "## References" + parts[1].strip() if len(parts) > 1 else "## References\nNone provided"
#     return content, references


@app.route('/api/youtube_chat', methods=['POST'])
def youtube_chat():
    question = request.json.get('question')
    channels = request.json.get('channels', [])
    if not question:
        return jsonify({"error": "No question provided"}), 400

    def generate():
        try:
            retriever = create_youtube_retriever(channels=channels,
                                                embeddings=embeddings,
                                                tidb_connection_string=tidb_connection_string,
                                                YOUTUBE_TABLE_NAME=YOUTUBE_TABLE_NAME)
            chatbot_chain = create_chatbot(retriever=retriever,google_api_key=google_api_key)
            response = chatbot_chain.invoke(question)
            
            content, references = split_response(response.content)
            
            for chunk in content.split('\n'):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            if references:
                yield f"data: {json.dumps({'references': references})}\n\n"
            
        except Exception as e:
            logging.exception("Error in YouTube chatbot response generation")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"

    return Response(generate(), content_type='text/event-stream')


@app.route('/api/player_list', methods=['GET'])
def get_player_list():
    player_list = [player['player'] for player in draft_assistant.players]
    return jsonify(player_list)


@app.route('/api/similar_players', methods=['POST'])
def similar_players():
    player_name = request.json.get('player')
    if not player_name:
        return jsonify({"error": "No player name provided"}), 400

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_api_key)
        vector_store = TiDBVectorStore.from_existing_vector_table(
            embedding=embeddings,
            connection_string=tidb_connection_string,
            table_name=PLAYER_REPORT_TABLE_NAME
        )

        # First, find the exact player
        player_docs = vector_store.similarity_search_with_score(player_name, k=1)
        if not player_docs:
            return jsonify({"error": f"Player '{player_name}' not found"}), 404

        player_doc, _ = player_docs[0]

        # Now, use the player's document to find similar players
        similar_docs = vector_store.similarity_search_with_score(player_doc.page_content, k=7)  # Get 7 to account for the original player
        
        results = []
        for doc, score in similar_docs:
            if doc.metadata['player'] != player_name:  # Exclude the queried player
                results.append({
                    "player": doc.metadata['player'],
                    "position": doc.metadata['pos'],
                    "ppr_projection": doc.metadata['ppr_projection'],
                    "outlook": doc.page_content,
                    "similarity_score": 1 - score  # Convert distance to similarity
                })

        # Sort by similarity score (highest first) and take top 6
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return jsonify(results[:6])  # Return top 6 similar players

    except Exception as e:
        logging.exception("Error in similar players search")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)