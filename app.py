import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="IPL Insights Dashboard", layout="wide")

# Title
st.title("🏏 IPL Insights Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("ipl-matches.csv")

df = load_data()

# Display first few rows
st.subheader("📄 First 10 Rows of the Dataset")
st.dataframe(df.head(10))

# Most Successful Teams
st.subheader("🏆 Most Successful Teams")
team_wins = df['WinningTeam'].value_counts()
st.bar_chart(team_wins)

# Most Player of the Match Awards
st.subheader("🥇 Top 10 Players of the Match")
top_players = df['Player_of_Match'].value_counts().head(10)
fig1, ax1 = plt.subplots()
top_players.plot(kind='bar', color='orange', ax=ax1)
ax1.set_xlabel("Player")
ax1.set_ylabel("Awards")
ax1.set_title("Top 10 Players of the Match")
st.pyplot(fig1)

# Venues with Most Matches
st.subheader("🏟️ Top 5 Venues by Matches Hosted")
top_venues = df['Venue'].value_counts().head(5)
fig2, ax2 = plt.subplots()
top_venues.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_ylabel("")
ax2.set_title("Top 5 Venues")
st.pyplot(fig2)

# Matches Per Season
st.subheader("📅 Matches Per Season")
matches_per_season = df['Season'].value_counts().sort_index()
fig3, ax3 = plt.subplots()
matches_per_season.plot(kind='line', marker='o', ax=ax3)
ax3.set_xlabel("Season")
ax3.set_ylabel("Number of Matches")
ax3.set_title("Matches Played per IPL Season")
ax3.grid(True)
st.pyplot(fig3)

# Team Win Percentage
st.subheader("📊 Team Win Percentage")
total_matches = df['Team1'].value_counts() + df['Team2'].value_counts()
win_percentage = (team_wins / total_matches * 100).sort_values(ascending=False)
st.bar_chart(win_percentage)

# Most Consistent Players
st.subheader("🏅 Most Consistent Players")
consistent_players = df.groupby(['Season', 'Player_of_Match']).size().reset_index(name='Awards')
top_consistent = consistent_players.groupby('Player_of_Match')['Awards'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_consistent)

# Team Performance Over Seasons
st.subheader("📈 Team Performance Over Seasons")
team_season_wins = df.groupby(['Season', 'WinningTeam']).size().unstack(fill_value=0)
st.line_chart(team_season_wins.T)

# Predictive Modeling: Match Outcome
st.subheader("🤖 Match Outcome Prediction Accuracy")
df_model = df[['TossWinner', 'Venue', 'WinningTeam']].dropna()
le = LabelEncoder()
df_model = df_model.apply(le.fit_transform)

X = df_model[['TossWinner', 'Venue']]
y = df_model['WinningTeam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: **{accuracy:.2%}**")

# Interactive Team Filter
st.subheader("🔍 Match Records by Team")
selected_team = st.selectbox("Select a Team", sorted(df['Team1'].dropna().unique()))
filtered_df = df[(df['Team1'] == selected_team) | (df['Team2'] == selected_team)]
st.write(f"Matches involving **{selected_team}**")
st.dataframe(filtered_df[['Season', 'Team1', 'Team2', 'WinningTeam']])

