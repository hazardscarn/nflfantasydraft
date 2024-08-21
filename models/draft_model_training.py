import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import os
import random

class FantasyFootballDraftAssistant:
    def __init__(self):
        self.df = None
        self.positions = {'QB': (1, 2), 'RB': (6, 9), 'WR': (5, 9), 'TE': (1, 2), 'K': (1, 1), 'DST': (1, 1)}
        self.starting_positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1, 'FLEX': 1}
        self.flex_positions = ['RB', 'WR', 'TE']
        self.q_table = defaultdict(float)
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.total_episodes = 0
        self.base_exploration_N = 15  # Maximum exploration cap
        self.min_exploration_N = 5    # Minimum exploration, even in late rounds
        self.position_exploration_factor = {
            'QB': 0.7, 'RB': 1.2, 'WR': 1.2, 'TE': 0.9, 'K': 0.3, 'DST': 0.3
        }
        self.epsilon_start = 0.1  # Lower starting epsilon for fine-tuning
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.99995  # Slower decay
        self.epsilon = self.epsilon_start
        self.num_teams = 12

    def get_exploration_N(self, round_num, position):
        round_factor = max(0.7, 1.5 - (round_num - 1) * 0.05)  # Slower decrease, minimum 0.7
        position_factor = self.position_exploration_factor.get(position, 1.0)
        exploration_N = int(self.base_exploration_N * round_factor * position_factor)
        return max(self.min_exploration_N, min(exploration_N, self.base_exploration_N))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)  

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['ADP'] = pd.to_numeric(self.df['ADP'], errors='coerce')
        self.df = self.df.sort_values('ADP').reset_index(drop=True)
        self.players = self.df.to_dict('records')
        
    def get_state(self, team, round_num, pick_number):
        pos_counts = [sum(1 for p in team if p['pos'] == pos) for pos in self.positions.keys()]
        return tuple(pos_counts + [round_num, pick_number])
    
    def get_actions(self, available_players, team, round_num):
        actions = []
        for player in available_players:
            pos = player['pos']
            pos_count = sum(1 for p in team if p['pos'] == pos)
            if pos_count < self.positions[pos][1] and (pos not in ['K', 'DST'] or round_num > 12):
                actions.append(player['player'])
        return actions
    
    def calculate_team_value(self, team):
        weekly_scores = []
        for week in range(1, 18):  # 17-week season
            available_players = [p for p in team if p['bye_week'] != week]
            starters = self.select_starters(available_players)
            week_score = sum(p['ppr_projection'] / 17 for p in starters)
            weekly_scores.append(week_score)
        
        total_score = sum(weekly_scores)
        bench_strength = sum(p['ppr_projection'] for p in team) - sum(p['ppr_projection'] for p in self.select_starters(team))
        return total_score + (bench_strength * 0.1)  # Adding some value for bench strength
    
    def select_starters(self, available_players):
        starters = []
        for pos, count in self.starting_positions.items():
            if pos == 'FLEX':
                flex_options = sorted([p for p in available_players if p['pos'] in self.flex_positions and p not in starters], 
                                      key=lambda x: x['ppr_projection'], reverse=True)
                starters.extend(flex_options[:count])
            else:
                pos_players = sorted([p for p in available_players if p['pos'] == pos and p not in starters], 
                                     key=lambda x: x['ppr_projection'], reverse=True)
                starters.extend(pos_players[:count])
        return starters
    
    def calculate_reward(self, team, player, round_num):
        team_value_before = self.calculate_team_value(team)
        new_team = team + [player]
        team_value_after = self.calculate_team_value(new_team)
        value_added = team_value_after - team_value_before
        
        adp_bonus = max(0, (200 - player['ADP']) / 10) if not np.isnan(player['ADP']) else 0
        
        pos_counts = {pos: sum(1 for p in team if p['pos'] == pos) for pos in self.positions.keys()}
        
        # Adjust reward calculation for later rounds
        if round_num > 10:
            value_added *= 1.2  # Increase the importance of value added in later rounds
        
        # Prioritize key positions in early rounds
        if round_num <= 6:
            if player['pos'] == 'QB':
                if pos_counts['QB'] == 0:
                    value_added *= 1.5
                else:
                    value_added *= 0.5  # Penalize drafting a second QB early
            elif player['pos'] in ['RB', 'WR']:
                value_added *= 1.5
            elif player['pos'] == 'TE':
                if pos_counts['TE'] == 0:
                    value_added *= 1.5
                else:
                    value_added *= 0.5  # Penalize drafting a second TE early
        
        # Adjust penalty for K and DST
        if player['pos'] in ['K', 'DST']:
            if round_num <= 12:
                value_added *= 0.1
            else:
                value_added *= 0.8  # Less penalty in very late rounds
        
        # Boost value for filling starting lineup
        if pos_counts[player['pos']] < self.starting_positions.get(player['pos'], 0):
            value_added *= 1.5
        
        # Add a small bonus for drafting any player in later rounds
        late_round_bonus = max(0, (round_num - 10) * 0.5) if round_num > 10 else 0
        
        return value_added + adp_bonus + late_round_bonus
    
    def train(self, num_episodes=50000):
        start_episode = self.total_episodes
        for episode in range(start_episode, start_episode + num_episodes):
            if episode % 1000 == 0:
                print(f"Training episode {episode}/{start_episode + num_episodes}")
            
            user_position = random.randint(0, self.num_teams - 1)  # Randomize user position for each episode
            teams = [[] for _ in range(self.num_teams)]
            available_players = self.players.copy()
            
            for round_num in range(1, 19):
                draft_order = range(self.num_teams) if round_num % 2 == 1 else reversed(range(self.num_teams))
                for pick_in_round, team_index in enumerate(draft_order):
                    pick_number = (round_num - 1) * self.num_teams + pick_in_round + 1
                    state = self.get_state(teams[team_index], round_num, pick_number)
                    actions = self.get_actions(available_players, teams[team_index], round_num)
                    
                    if not actions:
                        continue
                    
                    if np.random.uniform(0, 1) < self.epsilon:
                        # Exploration: Choose randomly from top N ADP players
                        sorted_actions = sorted(actions, key=lambda a: next((p['ADP'] for p in available_players if p['player'] == a), float('inf')))
                        player = next(p for p in available_players if p['player'] == sorted_actions[0])
                        position = player['pos']
                        exploration_N = self.get_exploration_N(round_num, position)
                        top_N_adp = sorted_actions[:exploration_N]
                        action = np.random.choice(top_N_adp)
                    else:
                        # Exploitation: Choose the action with the highest Q-value
                        q_values = [self.q_table[(state, a)] for a in actions]
                        action = actions[np.argmax(q_values)]
                    
                    player = next(p for p in available_players if p['player'] == action)
                    reward = self.calculate_reward(teams[team_index], player, round_num)
                    teams[team_index].append(player)
                    available_players = [p for p in available_players if p['player'] != action]
                    
                    next_round = round_num + 1 if pick_in_round == self.num_teams - 1 else round_num
                    next_pick = (pick_number % self.num_teams) + 1
                    next_state = self.get_state(teams[team_index], next_round, next_pick)
                    
                    # Q-value update
                    max_future_q = max([self.q_table[(next_state, a)] for a in self.get_actions(available_players, teams[team_index], next_round)], default=0)
                    current_q = self.q_table[(state, action)]
                    new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
                    self.q_table[(state, action)] = new_q
            
            # Decay epsilon after each episode
            self.decay_epsilon()
        
        self.total_episodes += num_episodes
    
    def recommend_players(self, team, available_players, round_num, pick_number, num_recommendations=5):
        state = self.get_state(team, round_num, pick_number)
        actions = self.get_actions(available_players, team, round_num)
        q_values = [(a, self.q_table[(state, a)]) for a in actions]
        
        # Combine Q-values with ADP for ranking
        adp_values = [(a, next((p['ADP'] for p in available_players if p['player'] == a), float('inf'))) for a in actions]
        combined_values = [(a, q + max(0, (200 - adp) / 400)) for (a, q), (_, adp) in zip(q_values, adp_values)]
        
        top_actions = sorted(combined_values, key=lambda x: x[1], reverse=True)[:num_recommendations]
        player_dict = {p['player']: p for p in available_players}
        return [(player_dict[a], q) for a, q in top_actions]
    
    def simulate_draft(self, user_position):
        teams = [[] for _ in range(self.num_teams)]
        available_players = self.players.copy()
        user_picks = []
        
        for round_num in range(1, 19):
            draft_order = range(self.num_teams) if round_num % 2 == 1 else reversed(range(self.num_teams))
            for pick_in_round, team_index in enumerate(draft_order):
                pick_number = (round_num - 1) * self.num_teams + pick_in_round + 1
                if team_index == user_position:
                    recommendations = self.recommend_players(teams[team_index], available_players, round_num, pick_number)
                    user_picks.append((pick_number, recommendations))
                    
                    if recommendations:
                        selected_player, _ = recommendations[0]
                        teams[team_index].append(selected_player)
                        available_players = [p for p in available_players if p['player'] != selected_player['player']]
                else:
                    # Other teams draft based on ADP with round-based randomness
                    valid_players = [p for p in available_players if sum(1 for player in teams[team_index] if player['pos'] == p['pos']) < self.positions[p['pos']][1]]
                    if valid_players:
                        # Sort valid players by ADP, handling NaN values
                        sorted_players = sorted(valid_players, key=lambda p: p['ADP'] if not np.isnan(p['ADP']) else float('inf'))
                        
                        # Determine the number of top players to consider based on the round
                        if round_num <= 1:
                            top_n = 2  # Pick the top ADP in rounds 1 and 2
                        elif round_num <= 3:
                            top_n = 2  # Pick from top 2 ADP in round 3
                        elif 4 <= round_num <= 6:
                            top_n = 3  # Pick from top 3 ADP in rounds 4, 5, 6
                        else:
                            top_n = 5  # Pick from top 5 ADP in later rounds
                        
                        # Select top N players (or fewer if less than N are available)
                        top_players = sorted_players[:min(top_n, len(sorted_players))]
                        
                        # Randomly select one player from the top players
                        player = random.choice(top_players)
                        teams[team_index].append(player)
                        available_players = [p for p in available_players if p['player'] != player['player']]
        
        return user_picks, teams
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((dict(self.q_table), self.total_episodes), f)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_q_table, self.total_episodes = pickle.load(f)
            self.q_table = defaultdict(float, loaded_q_table)
        print(f"Model loaded from {file_path}. Total episodes: {self.total_episodes}")

if __name__ == '__main__':
    # Training the model
    assistant = FantasyFootballDraftAssistant()
    data_file = 'data/cbs_fantasy_projection_master.csv'
    print(f"Loading data from {data_file}...")
    assistant.load_data(data_file)

    model_file = 'models//fantasy_football_model.pkl'

    # Check if a saved model exists
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}...")
        assistant.load_model(model_file)
        print("Continuing training from saved model...")
    else:
        print("No existing model found. Starting new training...")

    # Train the model
    num_episodes = 140000
    print(f"Training the model for {num_episodes} episodes...")
    assistant.train(num_episodes=num_episodes)

    # Save the model after training
    print(f"Saving the trained model to {model_file}...")
    assistant.save_model(model_file)

    print("Training complete!")

    # Optional: You can add code here to test the model or run simulations
    # For example:
    user_position = 8  # Example user position
    user_picks, teams = assistant.simulate_draft(user_position)
    print(f"Draft simulation complete. User picks: {user_picks}")
