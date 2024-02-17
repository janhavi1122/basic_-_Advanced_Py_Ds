# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:59:51 2023

@author: santo
"""

# Task 1: Read the data set and replace dashes with 0
import pandas as pd

# Read the CSV file
df = pd.read_csv("C:/datasets/Laliga.csv")

# Replace dashes with 0
df = df.replace('-', 0)

# Task 2: Print teams which started playing between 1930-1980
df['Debut'] = df['Debut'].str[:4].astype(int) # Extract the first 4 characters and convert to integer
teams_between_1930_1980 = df[(df['Debut'] >= 1930) & (df['Debut'] <= 1980)]
print(teams_between_1930_1980['Team'])

# Task 3: Print the list of teams which came Top 5 in terms of points
df['Points'] = df['Points'].astype(int) # Convert Points to integer for sorting
top_5_teams = df.nlargest(5, 'Points')
print(top_5_teams['Team'])

# Task 4: Define the function "Goal_diff_count"
def Goal_diff_count(row):
    return int(row['GoalsFor']) - int(row['GoalsAgainst'])

# Apply the function and create a new column "GoalDiff"
df['GoalDiff'] = df.apply(Goal_diff_count, axis=1)

# Find team with maximum and minimum goal difference
max_goal_diff_team = df[df['GoalDiff'] == df['GoalDiff'].max()]['Team'].values[0]
min_goal_diff_team = df[df['GoalDiff'] == df['GoalDiff'].min()]['Team'].values[0]
print(f'Team with maximum goal difference: {max_goal_diff_team}')
print(f'Team with minimum goal difference: {min_goal_diff_team}')

# Task 5: Create a new column "Winning Percent"
df['GamesWon'] = df['GamesWon'].astype(int)
df['GamesPlayed'] = df['GamesPlayed'].astype(int)

df['WinningPercent'] = (df['GamesWon'] / df['GamesPlayed']) * 100
df['WinningPercent'].fillna(0, inplace=True) # Replace NaN with 0

# Print top 5 teams with highest Winning percentage
top_5_winning_percent_teams = df.nlargest(5, 'WinningPercent')
print(top_5_winning_percent_teams[['Team', 'WinningPercent']])

# Task 6: Group teams based on their "Best position" and print the sum of their points for all positions
grouped_data = df.groupby('BestPosition')['Points'].sum()
print(grouped_data)