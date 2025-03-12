import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Hide Streamlit menus and set a green football-inspired background
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            body {
                background-color: #e8f5e9;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# App Title & Description
st.title("Foosball Winner Predictor")
st.markdown("<h3 style='color: green;'>Predict the Winner of a Foosball Game</h3>", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("foosball_model.pkl")

model_pipeline = load_model()

# Load player stats
@st.cache_data
def get_player_stats():
    try:
        df = pd.read_csv("Foosball.csv")
        
        # Pre-defined list of player names
        all_players = ['Aart-Jan', 'Amir', 'Andres', 'Arian', 'Berkan', 'Bob', 'Cathleen', 
                       'Ellen', 'Hans', 'Ilhan', 'Isabella', 'Jakko', 'Jeffrey', 'Jeroen', 
                       'Keon', 'Lara', 'Lars', 'Lucas', 'Luuk', 'Mohammad', 'Norent', 
                       'Robin', 'Roel', 'Ruby', 'Salma', 'Sander']
        
        # Initialize player stats
        player_ratings = {player: 1500 for player in all_players}
        player_experience = {player: 0 for player in all_players}
        
        # Convert game date to datetime and sort
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        df = df.sort_values('gameDate')
        
        # Add winner column
        df['winner'] = df.apply(lambda row: 1 if row['team1Goals'] > row['team2Goals'] else 2, axis=1)
        
        # Update player stats based on game history
        for _, match in df.iterrows():
            p1, p2 = match['player1Name'], match['player2Name']
            p3, p4 = match['player3Name'], match['player4Name']
            winner = match['winner']
            
            # Update experience
            player_experience[p1] += 1
            player_experience[p2] += 1
            player_experience[p3] += 1
            player_experience[p4] += 1
            
            # Update ratings (Elo system)
            team1_avg_rating = (player_ratings[p1] + player_ratings[p2]) / 2
            team2_avg_rating = (player_ratings[p3] + player_ratings[p4]) / 2
            
            K = 32
            team1_expected = 1 / (1 + 10**((team2_avg_rating - team1_avg_rating) / 400))
            team2_expected = 1 - team1_expected
            
            team1_actual = 1 if winner == 1 else 0
            team2_actual = 1 - team1_actual
            
            team1_rating_change = K * (team1_actual - team1_expected)
            team2_rating_change = K * (team2_actual - team2_expected)
            
            team1_avg_exp = (player_experience[p1] + player_experience[p2]) / 2
            team2_avg_exp = (player_experience[p3] + player_experience[p4]) / 2
            
            exp_factor1 = 1 / (1 + 0.1 * team1_avg_exp)
            exp_factor2 = 1 / (1 + 0.1 * team2_avg_exp)
            
            team1_adj_change = team1_rating_change * exp_factor1
            team2_adj_change = team2_rating_change * exp_factor2
            
            player_ratings[p1] += team1_adj_change
            player_ratings[p2] += team1_adj_change
            player_ratings[p3] += team2_adj_change
            player_ratings[p4] += team2_adj_change
        
        min_date = df['gameDate'].min()
        
        return all_players, player_ratings, player_experience, min_date
    except Exception as e:
        st.error(f"Error loading player stats: {e}")
        default_players = ['Aart-Jan', 'Amir', 'Andres', 'Arian', 'Berkan', 'Bob', 'Cathleen', 
                       'Ellen', 'Hans', 'Ilhan', 'Isabella', 'Jakko', 'Jeffrey', 'Jeroen', 
                       'Keon', 'Lara', 'Lars', 'Lucas', 'Luuk', 'Mohammad', 'Norent', 
                       'Robin', 'Roel', 'Ruby', 'Salma', 'Sander']
        default_ratings = {player: 1500 for player in default_players}
        default_experience = {player: 0 for player in default_players}
        return default_players, default_ratings, default_experience, pd.Timestamp("2025-03-01")

players_list, player_ratings, player_experience, min_date = get_player_stats()

# Function to standardize team order
def standardize_team(player1, player2):
    return (player1, player2) if player1 < player2 else (player2, player1)

# Function to prepare features for prediction
def create_features_for_prediction(team1, team2, days_since_start):
    p1, p2 = standardize_team(team1[0], team1[1])
    p3, p4 = standardize_team(team2[0], team2[1])
    
    p1_rating, p2_rating = player_ratings[p1], player_ratings[p2]
    p3_rating, p4_rating = player_ratings[p3], player_ratings[p4]
    
    p1_exp, p2_exp = player_experience[p1], player_experience[p2]
    p3_exp, p4_exp = player_experience[p3], player_experience[p4]
    
    # Calculate win rates based on experience and ratings
    p1_win_rate = 0.5 if p1_exp == 0 else 0.5 + (p1_rating - 1500) / 1000
    p2_win_rate = 0.5 if p2_exp == 0 else 0.5 + (p2_rating - 1500) / 1000
    p3_win_rate = 0.5 if p3_exp == 0 else 0.5 + (p3_rating - 1500) / 1000
    p4_win_rate = 0.5 if p4_exp == 0 else 0.5 + (p4_rating - 1500) / 1000
    
    # Clamp win rates between 0.1 and 0.9
    p1_win_rate = max(0.1, min(0.9, p1_win_rate))
    p2_win_rate = max(0.1, min(0.9, p2_win_rate))
    p3_win_rate = max(0.1, min(0.9, p3_win_rate))
    p4_win_rate = max(0.1, min(0.9, p4_win_rate))
    
    team1_avg_rating = (p1_rating + p2_rating) / 2
    team2_avg_rating = (p3_rating + p4_rating) / 2
    team1_avg_exp = (p1_exp + p2_exp) / 2
    team2_avg_exp = (p3_exp + p4_exp) / 2
    team1_win_rate = (p1_win_rate + p2_win_rate) / 2
    team2_win_rate = (p3_win_rate + p4_win_rate) / 2
    
    rating_diff = team1_avg_rating - team2_avg_rating
    exp_diff = team1_avg_exp - team2_avg_exp
    win_rate_diff = team1_win_rate - team2_win_rate
    
    features = [
        p1_rating, p2_rating, p3_rating, p4_rating,
        p1_exp, p2_exp, p3_exp, p4_exp,
        p1_win_rate, p2_win_rate, p3_win_rate, p4_win_rate,
        team1_avg_rating, team2_avg_rating,
        rating_diff, exp_diff, win_rate_diff,
        days_since_start
    ]
    
    return np.array([features])

# Tabs for different app sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Player Stats", "Model Info"])

with tab1:
    # Team selections using multiselect
    st.subheader("Select Team 1 Players")
    team1 = st.multiselect("Team 1 (Select exactly 2 players)", players_list, key="team1")

    st.subheader("Select Team 2 Players")
    team2 = st.multiselect("Team 2 (Select exactly 2 players)", players_list, key="team2")

    # Date input for the game (default is today)
    game_date = st.date_input("Game Date", value=datetime.today())

    # Check that exactly two players are selected per team and all players are unique
    if len(team1) != 2:
        st.error("Please select exactly 2 players for Team 1.")
    elif len(team2) != 2:
        st.error("Please select exactly 2 players for Team 2.")
    elif set(team1).intersection(set(team2)):
        st.error("A player cannot be on both teams. Please select different players.")
    else:
        # Compute days since feature
        game_date_ts = pd.Timestamp(game_date)
        days_since = (game_date_ts - min_date).days
        
        # Get prediction features
        X_pred = create_features_for_prediction(team1, team2, days_since)
        
        if st.button("Predict Winner"):
            # Get prediction probabilities
            pred_proba = model_pipeline.predict_proba(X_pred)[0]
            team1_win_prob = pred_proba[0] * 100
            team2_win_prob = pred_proba[1] * 100
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            # Create team labels with player names
            team1_label = f"Team 1 ({team1[0]} & {team1[1]})"
            team2_label = f"Team 2 ({team2[0]} & {team2[1]})"
            
            # Display probabilities using two columns with custom formatting
            col1, col2 = st.columns(2)
            
            # Determine winner for visual indication
            if team1_win_prob > team2_win_prob:
                col1.markdown(f"<div style='padding: 10px; background-color: rgba(0, 255, 0, 0.2); border-radius: 5px;'><h3>{team1_label}</h3><h2>{team1_win_prob:.1f}%</h2><p>Predicted Winner</p></div>", unsafe_allow_html=True)
                col2.markdown(f"<div style='padding: 10px; background-color: #e06666; border-radius: 5px;'><h3>{team2_label}</h3><h2>{team2_win_prob:.1f}%</h2></div>", unsafe_allow_html=True)
                winner = team1_label
            else:
                col1.markdown(f"<div style='padding: 10px; background-color: #e06666; border-radius: 5px;'><h3>{team1_label}</h3><h2>{team1_win_prob:.1f}%</h2></div>", unsafe_allow_html=True)
                col2.markdown(f"<div style='padding: 10px; background-color: rgba(0, 255, 0, 0.2); border-radius: 5px;'><h3>{team2_label}</h3><h2>{team2_win_prob:.1f}%</h2><p>Predicted Winner</p></div>", unsafe_allow_html=True)
                winner = team2_label
            
            st.markdown(f"<h3 style='text-align: center; margin-top: 20px;'>🏆 {winner} is more likely to win! 🏆</h3>", unsafe_allow_html=True)
            
            # Show player stats influencing the prediction
            st.subheader("Player Stats Influencing Prediction")
            
            stats_data = []
            for player in team1 + team2:
                stats_data.append({
                    "Player": player,
                    "Team": "Team 1" if player in team1 else "Team 2",
                    "Rating": player_ratings[player],
                    "Experience": player_experience[player]
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

with tab2:
    st.subheader("Player Statistics")
    
    # Create a dataframe with all player stats
    player_stats_df = pd.DataFrame({
        'Player': list(player_ratings.keys()),
        'Rating': list(player_ratings.values()),
        'Experience': list(player_experience.values())
    }).sort_values('Rating', ascending=False)
    
    # Display the top players
    st.markdown("### Top Players by Rating")
    st.dataframe(player_stats_df.head(10), use_container_width=True)
    
    # Create a visualization
    st.markdown("### Player Skill vs Experience")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with colored points
    scatter = sns.scatterplot(data=player_stats_df, x='Experience', y='Rating', s=100, 
                             hue='Rating', palette='viridis', ax=ax)
    
    # Add player names as labels
    for _, row in player_stats_df.iterrows():
        ax.text(row['Experience'], row['Rating'], row['Player'], fontsize=9)
    
    ax.set_title('Player Skill vs Experience')
    ax.set_xlabel('Number of Games Played')
    ax.set_ylabel('Elo Rating')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Full player stats table
    st.markdown("### All Player Statistics")
    st.dataframe(player_stats_df, use_container_width=True)

with tab3:
    st.subheader("About the Prediction Model")
    
    st.markdown("""
    ### How the Model Works
    
    This foosball prediction model uses a combination of machine learning and Elo rating systems to predict match outcomes:
    
    1. **Player Skill Tracking**: Each player has an Elo rating that evolves over time based on match results
    
    2. **Experience Factor**: New players improve faster than veterans (implemented with an experience decay factor)
    
    3. **Feature Engineering**: The model considers:
       - Individual player ratings
       - Team composition and chemistry
       - Rating differences between teams
       - Experience levels
       - Historical win rates
    
    4. **Machine Learning Pipeline**: Standardizes the input data and uses a Random Forest classifier to predict outcomes
    
    ### Tips for Making Good Teams
    
    - **Balance experience**: Pair veteran players with newer players
    - **Consider player chemistry**: Some players perform better together
    - **Check recent performance**: Player ratings change over time
    
    ### Data Sources
    
    The model is trained on actual foosball match results recorded in the office.
    """)

# Footer with version info
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Foosball Predictor v2.0 | Updated: March 2025</p>", unsafe_allow_html=True)
