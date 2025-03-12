import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

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

# Pre-defined list of player names
players_list = ['Bob', 'Lucas', 'Robin', 'Salma', 'Keon', 'Ruby', 'Jeffrey', 
                'Luuk', 'Hans', 'Lars', 'Cathleen', 'Roel', 'Ilhan', 'Sander', 'Jakko', 
                'Berkan', 'Amir', 'Ellen', 'Arian', 'Lara', 'Aart-Jan', 'Mohammad', 
                'Andres', 'Norent', 'Isabella', 'Jeroen']

# Team selections using multiselect
st.subheader("Select Team 1 Players")
team1 = st.multiselect("Team 1 (Select exactly 2 players)", players_list)

st.subheader("Select Team 2 Players")
team2 = st.multiselect("Team 2 (Select exactly 2 players)", players_list)

# Date input for the game (default is today)
game_date = st.date_input("Game Date", value=datetime.today())

# Hard-coded minimum date (for feature calculation)
min_date = pd.Timestamp("2024-10-04 00:00:00")

# Check that exactly two players are selected per team
if len(team1) != 2:
    st.error("Please select exactly 2 players for Team 1.")
elif len(team2) != 2:
    st.error("Please select exactly 2 players for Team 2.")
else:
    # Compute the "days_since" feature based on the game date and the hard-coded min_date
    game_date_ts = pd.Timestamp(game_date)
    days_since = (game_date_ts - min_date).days

    # Suppose team1_input is a list of two player names for team1
    team1_sorted = sorted(team1)
    team2_sorted = sorted(team2)

    # Use these sorted lists in your DataFrame
    input_df = pd.DataFrame({
        'player1Name': [team1_sorted[0]],
        'player2Name': [team1_sorted[1]],
        'player3Name': [team2_sorted[0]],
        'player4Name': [team2_sorted[1]],
        'days_since': [days_since]
    })

    # Load the saved model (ensure that 'foosball_model_LR.pkl' is in the same directory)
    model = joblib.load("foosball_model_LR.pkl")

    # Get prediction probabilities (assumes model classes: 1 for Team 1 win, 0 for Team 2 win)
    pred_proba = model.predict_proba(input_df)[0]
    team1_win_prob = pred_proba[1] * 100
    team2_win_prob = pred_proba[0] * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Prediction Results")
    
    # Display probabilities using two columns
    col1, col2 = st.columns(2)
    col1.metric("Team 1 Win Probability", f"{team1_win_prob:.2f}%")
    col2.metric("Team 2 Win Probability", f"{team2_win_prob:.2f}%")
    
    # Provide a simple interpretation message
    if team1_win_prob > team2_win_prob:
        st.success("Team 1 is more likely to win!")
    else:
        st.success("Team 2 is more likely to win!")
