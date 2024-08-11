import pandas as pd
import numpy as np
import numba
from numba import jit, prange
import multiprocessing
import pickle
import os

@jit(nopython=True)
def calculate_team_value_numba(team, bye_weeks, projections, positions, starting_positions):
    weekly_scores = np.zeros(17)
    for week in range(17):
        available_players = [i for i, p in enumerate(team) if bye_weeks[i] != week + 1]
        starters = select_starters_numba(available_players, projections, positions, starting_positions)
        weekly_scores[week] = np.sum(projections[starters]) / 17
    
    total_score = np.sum(weekly_scores)
    bench_strength = np.sum(projections[team]) - np.sum(projections[starters])
    return total_score + (bench_strength * 0.1)

@jit(nopython=True)
def select_starters_numba(available_players, projections, positions, starting_positions):
    starters = []
    for pos, count in starting_positions.items():
        if pos == 4:  # FLEX
            flex_options = sorted([(i, projections[i]) for i in available_players if positions[i] in [1, 2, 3] and i not in starters],
                                  key=lambda x: x[1], reverse=True)
            starters.extend([x[0] for x in flex_options[:count]])
        else:
            pos_players = sorted([(i, projections[i]) for i in available_players if positions[i] == pos and i not in starters],
                                 key=lambda x: x[1], reverse=True)
            starters.extend([x[0] for x in pos_players[:count]])
    return starters


@jit(nopython=True)
def calculate_reward_numba(team, player, round_num, adp, projections, bye_weeks, positions, starting_positions):
    team_value_before = calculate_team_value_numba(team, bye_weeks, projections, positions, starting_positions)
    new_team = np.append(team, player)
    team_value_after = calculate_team_value_numba(new_team, bye_weeks, projections, positions, starting_positions)
    value_added = team_value_after - team_value_before
    
    adp_bonus = max(0, (200 - adp[player]) / 10) if not np.isnan(adp[player]) else 0
    pos_counts = np.bincount(positions[team], minlength=7)
    
    # Prioritize key positions in early rounds
    if round_num <= 6:
        if positions[player] == 0:  # QB
            if pos_counts[0] == 0:
                value_added *= 1.5
            else:
                value_added *= 0.5  # Penalize drafting a second QB early
        elif positions[player] in [1, 2]:  # RB, WR
            value_added *= 1.5
        elif positions[player] == 3:  # TE
            if pos_counts[3] == 0:
                value_added *= 1.5
            else:
                value_added *= 1  # Penalize drafting a second TE early
    
    # Penalize drafting K and DST early
    if positions[player] in [5, 6] and round_num <= 12:  # K, DST
        value_added *= 0.1
    
    # Boost value for filling starting lineup
    if pos_counts[positions[player]] < starting_positions[positions[player]]:
        value_added *= 1.5
    
    return value_added + adp_bonus

class FantasyFootballDraftAssistant:
    def __init__(self):
        self.df = None
        self.positions = {'QB': (1, 2), 'RB': (6, 9), 'WR': (5, 9), 'TE': (1, 2), 'K': (1, 1), 'DST': (1, 1)}
        self.starting_positions = {0: 1, 1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 1}  # QB, RB, WR, TE, FLEX, K, DST
        self.flex_positions = [1, 2, 3]  # RB, WR, TE
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.total_episodes = 0
        
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['ADP'] = pd.to_numeric(self.df['ADP'], errors='coerce')
        self.df = self.df.sort_values('ADP').reset_index(drop=True)
        self.players = self.df['player'].tolist()
        self.adp = self.df['ADP'].values
        self.projections = self.df['ppr_projection'].values
        self.bye_weeks = self.df['bye_week'].values
        self.positions_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'K': 5, 'DST': 6}
        self.positions_array = np.array([self.positions_map[pos] for pos in self.df['pos']])
        
    def get_state(self, team, round_num):
        pos_counts = np.bincount([self.positions_map[self.df.loc[self.df['player'] == player, 'pos'].iloc[0]] for player in team], minlength=7)
        return tuple(pos_counts.tolist() + [round_num])
    
    def get_actions(self, available_players, team, round_num):
        pos_counts = np.bincount([self.positions_map[self.df.loc[self.df['player'] == player, 'pos'].iloc[0]] for player in team], minlength=7)
        max_counts = np.array([2, 9, 9, 2, 0, 1, 1])
        valid_positions = (pos_counts < max_counts)[self.positions_array[self.df['player'].isin(available_players)]] & (
            (self.positions_array[self.df['player'].isin(available_players)] != 5) & 
            (self.positions_array[self.df['player'].isin(available_players)] != 6) | 
            (round_num > 12)
        )
        return np.array(available_players)[valid_positions]

    @jit(nopython=True)
    def train_episode_numba(self, players, adp, projections, bye_weeks, positions, starting_positions, q_table, alpha, gamma, epsilon):
        num_teams = 12
        teams = [np.array([], dtype=np.int64) for _ in range(num_teams)]
        available_players = players.copy()
        
        for round_num in range(1, 19):
            for team_index in range(num_teams):
                state = self.get_state(teams[team_index], round_num)
                actions = self.get_actions(available_players, teams[team_index], round_num)
                
                if len(actions) == 0:
                    continue
                
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.choice(actions)
                else:
                    q_values = np.array([q_table.get((state, a), 0) for a in actions])
                    action = actions[np.argmax(q_values)]
                
                reward = calculate_reward_numba(teams[team_index], action, round_num, adp, projections, bye_weeks, positions, starting_positions)
                teams[team_index] = np.append(teams[team_index], action)
                available_players = np.array([p for p in available_players if p != action])
                
                next_state = self.get_state(teams[team_index], round_num + 1)
                next_actions = self.get_actions(available_players, teams[team_index], round_num + 1)
                
                if len(next_actions) > 0:
                    max_future_q = np.max([q_table.get((next_state, a), 0) for a in next_actions])
                else:
                    max_future_q = 0
                
                current_q = q_table.get((state, action), 0)
                new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
                q_table[(state, action)] = new_q
        
        return q_table
    
    def train(self, num_episodes=10000):
        return self.train_cpu(num_episodes)
    
    def train_cpu(self, num_episodes):
        num_cores = multiprocessing.cpu_count()
        episodes_per_core = num_episodes // num_cores
        
        with multiprocessing.Pool(num_cores) as pool:
            results = pool.starmap(self.train_batch, [(episodes_per_core,) for _ in range(num_cores)])
        
        for result in results:
            for state_action, q_value in result.items():
                if state_action in self.q_table:
                    self.q_table[state_action] = (self.q_table[state_action] + q_value) / 2
                else:
                    self.q_table[state_action] = q_value
        
        self.total_episodes += num_episodes
    
    def train_batch(self, num_episodes):
        local_q_table = {}
        for _ in range(num_episodes):
            local_q_table = self.train_episode_numba(
                self.players, self.adp, self.projections, self.bye_weeks,
                self.positions_array, self.starting_positions, local_q_table,
                self.alpha, self.gamma, self.epsilon
            )
        return local_q_table
    
    # def recommend_players(self, team, available_players, round_num, num_recommendations=5):
    #     state = self.get_state(team, round_num)
    #     actions = self.get_actions(available_players, team, round_num)
    #     q_values = np.array([self.q_table.get((state, a), 0) for a in actions])
    #     top_indices = q_values.argsort()[-num_recommendations:][::-1]
    #     return [(self.df.iloc[actions[i]], q_values[i]) for i in top_indices]


    def recommend_players(self, team, available_players, round_num, num_recommendations=5):
        try:
            state = self.get_state(team, round_num)
            actions = self.get_actions(available_players, team, round_num)
            if len(actions) == 0:
                return [{"error": "No valid actions available", "details": {
                    "team": team,
                    "available_players": available_players,
                    "round_num": round_num
                }}]
            q_values = np.array([self.q_table.get((state, a), 0) for a in actions])
            top_indices = q_values.argsort()[-num_recommendations:][::-1]
            return [self.df.loc[self.df['player'] == actions[i]].to_dict('records')[0] for i in top_indices]
        except Exception as e:
            print(f"Error in recommend_players: {str(e)}")
            return [{"error": str(e), "details": {
                "team": team,
                "available_players": available_players,
                "round_num": round_num
            }}]
        
        
    def simulate_draft(self, user_position, num_teams=12):
        teams = [np.array([], dtype=np.int64) for _ in range(num_teams)]
        available_players = self.players.copy()
        user_picks = []
        
        for round_num in range(1, 19):
            draft_order = range(num_teams) if round_num % 2 == 1 else reversed(range(num_teams))
            for pick_in_round, team_index in enumerate(draft_order):
                if team_index == user_position:
                    recommendations = self.recommend_players(teams[team_index], available_players, round_num)
                    user_picks.append(((round_num - 1) * num_teams + pick_in_round + 1, recommendations))
                    
                    if recommendations:
                        selected_player = recommendations[0][0].name
                        teams[team_index] = np.append(teams[team_index], selected_player)
                        available_players = np.array([p for p in available_players if p != selected_player])
                else:
                    valid_players = self.get_actions(available_players, teams[team_index], round_num)
                    if len(valid_players) > 0:
                        player = valid_players[np.argmin(self.adp[valid_players])]
                        teams[team_index] = np.append(teams[team_index], player)
                        available_players = np.array([p for p in available_players if p != player])
        
        return user_picks, teams
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.q_table, self.total_episodes), f)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.q_table, self.total_episodes = pickle.load(f)
        print(f"Model loaded from {file_path}. Total episodes: {self.total_episodes}")
