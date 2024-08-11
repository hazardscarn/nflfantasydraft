import pandas as pd
import numpy as np

class FantasyFootballDraftAssistant:
    def __init__(self):
        self.df = None
        self.positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1, 'FLEX': 1}
        self.flex_positions = ['RB', 'WR', 'TE']
        
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['ADP'] = pd.to_numeric(self.df['ADP'], errors='coerce')
        self.df = self.df.sort_values('ppr_projection', ascending=False).reset_index(drop=True)
        
    def calculate_score(self, player, draft_position, team):
        score = player['ppr_projection']
        
        # Adjust score based on ADP vs current draft position
        if not np.isnan(player['ADP']):
            adp_diff = draft_position - player['ADP']
            score += adp_diff * 0.5  # Boost score if player is available later than expected
        
        # Position need
        pos_count = sum(1 for p in team if p['pos'] == player['pos'])
        if pos_count < self.positions.get(player['pos'], 0):
            score += 20  # Boost score if position need isn't met
        elif player['pos'] in self.flex_positions and pos_count < self.positions.get(player['pos'], 0) + self.positions['FLEX']:
            score += 10  # Smaller boost for FLEX-eligible positions
        
        # Bench strength
        if pos_count >= self.positions.get(player['pos'], 0):
            score += 5  # Small boost for improving bench strength
        
        # Bye week coverage
        if any(p['bye_week'] == player['bye_week'] for p in team if p['pos'] == player['pos']):
            score -= 5  # Penalize for overlapping bye weeks
        
        return score
        
    def recommend_players(self, draft_position, team, num_recommendations=5):
        available_players = self.df[~self.df['player'].isin([p['player'] for p in team])]
        
        scores = [(player, self.calculate_score(player, draft_position, team)) 
                  for _, player in available_players.iterrows()]
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    def simulate_draft(self, num_teams=12, num_rounds=18):
        draft_order = list(range(num_teams)) * num_rounds
        for i in range(1, num_rounds, 2):
            draft_order[i*num_teams:(i+1)*num_teams] = reversed(draft_order[i*num_teams:(i+1)*num_teams])
        
        teams = [[] for _ in range(num_teams)]
        
        for pick, team_index in enumerate(draft_order):
            draft_position = pick + 1
            recommendations = self.recommend_players(draft_position, teams[team_index])
            selected_player = recommendations[0][0]
            teams[team_index].append(selected_player)
            self.df = self.df[self.df['player'] != selected_player['player']]
        
        return teams

# Usage and testing
assistant = FantasyFootballDraftAssistant()
assistant.load_data('cbs_fantasy_projection_master.csv')

# Test recommendations
test_team = []
recommendations = assistant.recommend_players(1, test_team)
print("Top 5 recommendations for the first pick:")
for player, score in recommendations:
    print(f"{player['player']} ({player['pos']}) - Score: {score:.2f}")

# Simulate a draft
simulated_teams = assistant.simulate_draft()
print("\nFirst three picks for each team:")
for i, team in enumerate(simulated_teams):
    print(f"Team {i+1}: {', '.join([f'{p['player']} ({p['pos']})' for p in team[:3]])}")

# Analyze position distribution
def analyze_team(team):
    positions = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0, 'DST': 0}
    for player in team:
        positions[player['pos']] += 1
    return positions

print("\nPosition distribution for each team:")
for i, team in enumerate(simulated_teams):
    positions = analyze_team(team)
    print(f"Team {i+1}: {positions}")