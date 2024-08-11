import pandas as pd
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.DEBUG)

class FantasyFootballDraftAssistant:
    def __init__(self):
        self.df = None
        self.positions = {'QB': (1, 2), 'RB': (6, 9), 'WR': (5, 9), 'TE': (1, 2), 'K': (1, 1), 'DST': (1, 1)}
        self.starting_positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1, 'FLEX': 1}
        self.flex_positions = ['RB', 'WR', 'TE']
        self.q_table = {}
        self.total_episodes = 0
        
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['ADP'] = pd.to_numeric(self.df['ADP'], errors='coerce')
        self.df = self.df.sort_values('ADP').reset_index(drop=True)
        self.players = self.df.to_dict('records')
        
    def get_state(self, team, round_num):
        team_dict = {player: self.df[self.df['player'] == player].iloc[0].to_dict() for player in team}
        return (tuple(sum(1 for p in team_dict.values() if p['pos'] == pos) for pos in self.positions.keys()), round_num)
    
    def get_actions(self, available_players, team, round_num):
        team_dict = {player: self.df[self.df['player'] == player].iloc[0].to_dict() for player in team}
        actions = []
        for player in available_players:
            player_data = self.df[self.df['player'] == player].iloc[0]
            pos = player_data['pos']
            pos_count = sum(1 for p in team_dict.values() if p['pos'] == pos)
            if pos_count < self.positions[pos][1] and (pos not in ['K', 'DST'] or round_num > 12):
                actions.append(player)
        return actions
    
    def recommend_players(self, team, available_players, round_num, num_recommendations=5):
        try:
            state = self.get_state(team, round_num)
            actions = self.get_actions(available_players, team, round_num)
            logging.debug(f"State: {state}, Available actions: {actions}")
            logging.debug(f"Q-table size: {len(self.q_table)}")
            
            q_values = [(a, self.q_table.get((state, a), 0)) for a in actions]
            logging.debug(f"Q-values: {q_values}")
            
            top_actions = sorted(q_values, key=lambda x: x[1], reverse=True)[:num_recommendations]
            logging.debug(f"Top actions: {top_actions}")
            
            recommendations = []
            for a, q in top_actions:
                player = self.df[self.df['player'] == a].iloc[0].to_dict()
                player['q_value'] = q
                recommendations.append(player)
            
            logging.debug(f"Recommendations: {recommendations}")
            return recommendations
        except Exception as e:
            logging.exception(f"Error in recommend_players: {str(e)}")
            return [{"error": str(e), "details": {
                "team": team,
                "available_players": available_players,
                "round_num": round_num
            }}]
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_q_table, self.total_episodes = pickle.load(f)
            self.q_table = loaded_q_table
        logging.info(f"Model loaded from {file_path}. Total episodes: {self.total_episodes}")
        logging.debug(f"Q-table size after loading: {len(self.q_table)}")
# # For testing purposes
# if __name__ == "__main__":
#     assistant = FantasyFootballDraftAssistant()
#     assistant.load_data('data/cbs_fantasy_projection_master.csv')
#     model_file = 'data/fantasy_football_model.pkl'
    
#     if os.path.exists(model_file):
#         assistant.load_model(model_file)
#     else:
#         print("Model file not found. Please ensure the model is trained and saved.")
    
#     # Test recommendations
#     test_team = []
#     test_available_players = assistant.players[:20]  # Use first 20 players for testing
#     test_round = 1
    
#     recommendations = assistant.recommend_players(test_team, test_available_players, test_round)
#     print("\nTest Recommendations:")
#     for i, player in enumerate(recommendations, 1):
#         print(f"{i}. {player['player']} ({player['pos']}) - Q-value: {player['q_value']:.2f}, ADP: {player['ADP']:.2f}, Projection: {player['ppr_projection']:.2f}")