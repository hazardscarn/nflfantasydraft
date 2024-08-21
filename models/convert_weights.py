import sys
import os

# Add the path to your project directory
sys.path.append('D:/Work/Github/nflfantasydraft/models')

from draft_model_training_DQN import FantasyFootballDraftAssistant



def emergency_save():
    assistant = FantasyFootballDraftAssistant()
    
    # Recreate the state of your assistant
    data_file = 'data/cbs_fantasy_projection_master.csv'
    assistant.load_data(data_file)
    
    # Set the epsilon to the final value from your training
    assistant.dqn.epsilon = 0.0100
    
    # Now try to save the model
    model_file = 'models/fantasy_football_dqn_emergency_save'
    assistant.save_model(model_file)

if __name__ == "__main__":
    emergency_save()