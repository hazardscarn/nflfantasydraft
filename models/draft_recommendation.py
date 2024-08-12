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
        self.num_teams = 12
        
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['ADP'] = pd.to_numeric(self.df['ADP'], errors='coerce')
        self.df = self.df.sort_values('ADP').reset_index(drop=True)
        self.players = self.df.to_dict('records')
        
    def get_state(self, team, round_num, pick_number):
        pos_counts = {pos: sum(1 for p in team if p['pos'] == pos) for pos in self.positions.keys()}
        state = tuple(list(pos_counts.values()) + [round_num, pick_number])
        logging.debug(f"Generated state: {state}")
        return state
    
    def get_actions(self, available_players, team, round_num):
        actions = []
        for player in available_players:
            pos = player['pos']
            pos_count = sum(1 for p in team if p['pos'] == pos)
            if pos_count < self.positions[pos][1] and (pos not in ['K', 'DST'] or round_num > 12):
                actions.append(player['player'])
        return actions
    
    def recommend_players(self, team, available_players, round_num, pick_number, num_recommendations=6):
        try:
            state = self.get_state(team, round_num, pick_number)
            actions = self.get_actions(available_players, team, round_num)
            logging.debug(f"State: {state}, Available actions: {actions}")
            logging.debug(f"Q-table size: {len(self.q_table)}")
            
            # Get Q-values for all available actions
            q_values = [(a, self.q_table.get((state, a), 0)) for a in actions]
            logging.debug(f"Q-values: {q_values}")
            
            # Sort actions by Q-value in descending order and take the top 'num_recommendations'
            top_actions = sorted(q_values, key=lambda x: x[1], reverse=True)[:num_recommendations]
            logging.debug(f"Top actions: {top_actions}")
            
            recommendations = []
            for a, q in top_actions:
                player = next((p for p in available_players if p['player'] == a), None)
                if player:
                    player_copy = player.copy()
                    player_copy['q_value'] = q
                    recommendations.append(player_copy)
            
            logging.debug(f"Recommendations: {recommendations}")
            return recommendations
        except Exception as e:
            logging.exception(f"Error in recommend_players: {str(e)}")
            return [{"error": str(e), "details": {
                "team": team,
                "available_players": available_players,
                "round_num": round_num,
                "pick_number": pick_number
            }}]
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.q_table, self.total_episodes = pickle.load(f)
        logging.info(f"Model loaded from {file_path}. Total episodes: {self.total_episodes}")
        logging.debug(f"Q-table size after loading: {len(self.q_table)}")
        logging.debug(f"Sample Q-table entries: {list(self.q_table.items())[:5]}")

# The simulate_draft function has been removed as it's not used