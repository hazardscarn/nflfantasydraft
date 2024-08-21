import os
import sys
import tensorflow as tf
import pickle
import numpy as np

# Add the path to your project directory
sys.path.append('D:/Work/Github/nflfantasydraft/models')

from draft_model_training_DQN import FantasyFootballDraftAssistant, DQN



def load_model(assistant, model_path):
    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} does not exist.")
        return False

    try:
        if model_path.endswith('.h5'):
            print("Detected .h5 file. Attempting to load full model...")
            assistant.dqn.model = tf.keras.models.load_model(model_path)
        elif model_path.endswith('.weights.h5'):
            print("Detected .weights.h5 file. Attempting to load weights...")
            assistant.dqn.model.load_weights(model_path)
        elif model_path.endswith('.pkl'):
            print("Detected .pkl file. Attempting to load pickled weights...")
            with open(model_path, 'rb') as f:
                weights = pickle.load(f)
            assistant.dqn.model.set_weights(weights)
        else:
            print("No recognized file extension. Attempting to load as a full model...")
            assistant.dqn.model = tf.keras.models.load_model(model_path)

        print(f"Model successfully loaded from {model_path}")
        return True

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Attempting alternative loading methods...")

        try:
            print("Trying to load as weights...")
            assistant.dqn.model.load_weights(model_path)
            print("Successfully loaded as weights.")
            return True
        except:
            print("Failed to load as weights.")

        try:
            print("Trying to load as pickled data...")
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                print("Loaded pickled data appears to be weights. Applying to model...")
                assistant.dqn.model.set_weights(data)
                print("Successfully applied pickled weights to model.")
                return True
            else:
                print("Loaded pickled data is not in the expected format.")
        except:
            print("Failed to load as pickled data.")

        print("All loading attempts failed.")
        return False

def run_simulation(assistant, user_position):
    print(f"Starting draft simulation. You are drafting at position {user_position}.")
    teams = [[] for _ in range(assistant.num_teams)]
    available_players = assistant.players.copy()
    
    for round_num in range(1, 19):
        print(f"\nRound {round_num}")
        draft_order = range(assistant.num_teams) if round_num % 2 == 1 else reversed(range(assistant.num_teams))
        for pick_in_round, team_index in enumerate(draft_order):
            pick_number = (round_num - 1) * assistant.num_teams + pick_in_round + 1
            
            if team_index == user_position:
                print(f"\nYour pick (Pick {pick_number}):")
                recommendations = assistant.recommend_players(teams[team_index], available_players, round_num, pick_number)
                for i, (player, q_value) in enumerate(recommendations, 1):
                    print(f"{i}. {player['player']} ({player['pos']}) - Q-value: {q_value:.4f}")
                
                choice = int(input("Enter the number of the player you want to draft: ")) - 1
                selected_player = recommendations[choice][0]
            else:
                state = assistant.get_state(teams[team_index], round_num, pick_number, available_players)
                available_actions = [i for i, p in enumerate(assistant.players) if p in available_players]
                action = assistant.dqn.act(state, available_actions)
                selected_player = assistant.players[action]
            
            teams[team_index].append(selected_player)
            available_players.remove(selected_player)
            
            print(f"Team {team_index + 1} drafted: {selected_player['player']} ({selected_player['pos']})")
    
    print("\nDraft complete! Here's your team:")
    for player in teams[user_position]:
        print(f"{player['player']} ({player['pos']})")

if __name__ == "__main__":
    assistant = FantasyFootballDraftAssistant()
    data_file = 'data/cbs_fantasy_projection_master.csv'
    assistant.load_data(data_file)
    
    model_file = 'models/fantasy_football_dqn_emergency_save'  # Update this path if needed
    load_model(assistant, model_file)
    
    user_position = int(input("Enter your draft position (0-11): "))
    run_simulation(assistant, user_position)