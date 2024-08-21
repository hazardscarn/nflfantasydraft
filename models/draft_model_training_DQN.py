import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
import random
import pickle
import os
import time

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        act_values = self.model.predict(state)
        return available_actions[np.argmax(act_values[0][available_actions])]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class FantasyFootballDraftAssistant:
    def __init__(self):
        self.positions = {'QB': (1, 2), 'RB': (6, 9), 'WR': (5, 9), 'TE': (1, 2), 'K': (1, 1), 'DST': (1, 1)}
        self.starting_positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1, 'FLEX': 1}
        self.flex_positions = ['RB', 'WR', 'TE']
        self.num_teams = 12
        self.state_size = 15  # Update this to match the actual state size
        self.action_size = 300  # Assuming 300 draftable players
        self.players = None
        self.df = None

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['ADP'] = pd.to_numeric(self.df['ADP'], errors='coerce')
        self.df = self.df.sort_values('ADP').reset_index(drop=True)
        self.players = self.df.to_dict('records')
        self.action_size = len(self.players)
        self.dqn = DQN(self.state_size, self.action_size)

    def get_state(self, team, round_num, pick_number, available_players):
        pos_counts = [sum(1 for p in team if p['pos'] == pos) / self.positions[pos][1] for pos in self.positions]
        avg_proj_points = [np.mean([p['ppr_projection'] for p in team if p['pos'] == pos]) if any(p['pos'] == pos for p in team) else 0 for pos in self.positions]
        remaining_players = len(available_players) / len(self.players)
        state = np.array(pos_counts + avg_proj_points + [remaining_players, round_num/18, pick_number/self.num_teams])
        return state.reshape(1, self.state_size)

    def calculate_reward(self, team, player, round_num):
        team_value_before = self.calculate_team_value(team)
        new_team = team + [player]
        team_value_after = self.calculate_team_value(new_team)
        value_added = team_value_after - team_value_before
        
        adp_bonus = max(0, (200 - player['ADP']) / 10) if not np.isnan(player['ADP']) else 0
        
        pos_counts = {pos: sum(1 for p in team if p['pos'] == pos) for pos in self.positions.keys()}
        
        if round_num > 10:
            value_added *= 1.2
        
        if round_num <= 6:
            if player['pos'] == 'QB':
                value_added *= 1.5 if pos_counts['QB'] == 0 else 0.5
            elif player['pos'] in ['RB', 'WR']:
                value_added *= 1.5
            elif player['pos'] == 'TE':
                value_added *= 1.5 if pos_counts['TE'] == 0 else 0.5
        
        if player['pos'] in ['K', 'DST']:
            value_added *= 0.1 if round_num <= 12 else 0.8
        
        if pos_counts[player['pos']] < self.starting_positions.get(player['pos'], 0):
            value_added *= 1.5
        
        late_round_bonus = max(0, (round_num - 10) * 0.5) if round_num > 10 else 0
        
        return value_added + adp_bonus + late_round_bonus

    def calculate_team_value(self, team):
        weekly_scores = []
        for week in range(1, 18):
            available_players = [p for p in team if p['bye_week'] != week]
            starters = self.select_starters(available_players)
            week_score = sum(p['ppr_projection'] / 17 for p in starters)
            weekly_scores.append(week_score)
        
        total_score = sum(weekly_scores)
        bench_strength = sum(p['ppr_projection'] for p in team) - sum(p['ppr_projection'] for p in self.select_starters(team))
        return total_score + (bench_strength * 0.1)

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



    def train(self, num_episodes=50000):
        batch_size = 32
        total_start_time = time.time()
        for episode in range(num_episodes):
            episode_start_time = time.time()
            teams = [[] for _ in range(self.num_teams)]
            available_players = self.players.copy()
            
            for round_num in range(1, 19):
                draft_order = range(self.num_teams) if round_num % 2 == 1 else reversed(range(self.num_teams))
                for pick_in_round, team_index in enumerate(draft_order):
                    pick_number = (round_num - 1) * self.num_teams + pick_in_round + 1
                    state = self.get_state(teams[team_index], round_num, pick_number, available_players)
                    
                    available_actions = [i for i, p in enumerate(self.players) if p in available_players]
                    action = self.dqn.act(state, available_actions)
                    
                    player = self.players[action]
                    reward = self.calculate_reward(teams[team_index], player, round_num)
                    teams[team_index].append(player)
                    available_players.remove(player)
                    
                    next_state = self.get_state(teams[team_index], round_num, pick_number + 1, available_players)
                    done = round_num == 18 and pick_in_round == self.num_teams - 1
                    
                    self.dqn.remember(state, action, reward, next_state, done)
                    
                    if len(self.dqn.memory) > batch_size:
                        self.dqn.replay(batch_size)
                    
                    if done:
                        break
                
                if done:
                    break
            
            if episode % 10 == 0:
                self.dqn.update_target_model()
            
            episode_time = time.time() - episode_start_time
            print(f"Episode: {episode}/{num_episodes}, Time: {episode_time:.2f}s, Epsilon: {self.dqn.epsilon:.4f}")
            # if episode % 100 == 0:
            #     print(f"Episode: {episode}/{num_episodes}, Time: {episode_time:.2f}s, Epsilon: {self.dqn.epsilon:.4f}")
        
        total_time = time.time() - total_start_time
        print(f"Training complete! Total time: {total_time/3600:.2f} hours")

    def recommend_players(self, team, available_players, round_num, pick_number, num_recommendations=5):
        state = self.get_state(team, round_num, pick_number, available_players)
        q_values = self.dqn.model.predict(state)[0]
        available_actions = [i for i, p in enumerate(self.players) if p in available_players]
        top_actions = sorted(available_actions, key=lambda x: q_values[x], reverse=True)[:num_recommendations]
        return [(self.players[action], q_values[action]) for action in top_actions]

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
                        available_players.remove(selected_player)
                else:
                    state = self.get_state(teams[team_index], round_num, pick_number, available_players)
                    available_actions = [i for i, p in enumerate(self.players) if p in available_players]
                    action = self.dqn.act(state, available_actions)
                    player = self.players[action]
                    teams[team_index].append(player)
                    available_players.remove(player)
        
        return user_picks, teams

    def save_model(self, file_path):
        try:
            # Try to save the entire model
            self.dqn.model.save(file_path)
            print(f"Full model saved to {file_path}")
        except Exception as e:
            print(f"Failed to save full model: {e}")
            try:
                # If that fails, try to save just the weights
                weights_path = file_path.rsplit('.', 1)[0] + '.weights.h5'
                self.dqn.model.save_weights(weights_path)
                print(f"Model weights saved to {weights_path}")
            except Exception as e:
                print(f"Failed to save model weights: {e}")
                # As a last resort, try to pickle the model
                pickle_path = file_path.rsplit('.', 1)[0] + '.pkl'
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.dqn.model.get_weights(), f)
                print(f"Model weights pickled to {pickle_path}")

    def load_model(self, file_path):
        self.dqn.load(file_path)
        print(f"Model loaded from {file_path}")

if __name__ == '__main__':
    assistant = FantasyFootballDraftAssistant()
    data_file = 'data//cbs_fantasy_projection_master.csv'
    print(f"Loading data from {data_file}...")
    assistant.load_data(data_file)

    model_file = 'models//fantasy_football_dqn.weights.h5'

    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}...")
        assistant.load_model(model_file)
        print("Continuing training from saved model...")
    else:
        print("No existing model found. Starting new training...")

    num_episodes = 10
    print(f"Training the model for {num_episodes} episodes...")
    assistant.train(num_episodes=num_episodes)

    print(f"Saving the trained model to {model_file}...")
    assistant.save_model(model_file)

    print("Training complete!")

    # Optional: Run a simulation
    user_position = 8  # Example user position
    user_picks, teams = assistant.simulate_draft(user_position)
    print(f"Draft simulation complete. User picks: {user_picks}")