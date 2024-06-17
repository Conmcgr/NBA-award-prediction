import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time
import os

team_abbreviations = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
        # Historical teams
        "Seattle SuperSonics": "SEA", "New Orleans Hornets": "NOH", "Charlotte Bobcats": "CHA",
        "Vancouver Grizzlies": "VAN", "San Diego Clippers": "SDC", "Kansas City Kings": "KCK",
        "Washington Bullets": "WSB", "Buffalo Braves": "BUF", "New Jersey Nets": "NJN",
        "New Orleans/Oklahoma City Hornets": "NOK", "St. Louis Hawks": "STL", 
        "Syracuse Nationals": "SYR", "Rochester Royals": "ROC", "Fort Wayne Pistons": "FWP",
        "Minneapolis Lakers": "MNL", "Cincinnati Royals": "CIN", "San Francisco Warriors": "SFW",
        "Philadelphia Warriors": "PHW", "Chicago Zephyrs": "CHZ", "Baltimore Bullets": "BAL",
        "Chicago Packers": "CHP", "Anderson Packers": "AND", "Sheboygan Red Skins": "SRS",
        "Waterloo Hawks": "WAT", "Tri-Cities Blackhawks": "TRI"
    }

def clean_stats(total_stats, wins):
    
    #Clean Wins and merge onto stats,
    wins['Team'] = wins['Team'].str.replace('*', '', regex=False)
    wins['Team'] = wins['Team'].map(team_abbreviations)
    wins = wins[['Rk','Team','W','L']]
    total_stats = total_stats.fillna(0)
    total_stats.drop(columns=['Rk'], inplace=True)
    total_stats = total_stats.merge(wins, left_on='Tm', right_on='Team', how='left')
    total_stats.drop(columns=['Team'], inplace=True)
    
    df = total_stats.copy()
    
    # Function to calculate weighted values for Rk, W, and L
    def calculate_weighted_values(row, total_games):
        if pd.notna(row['Rk']):
            weighted_rk = (int(row['G']) / total_games) * row['Rk']
        else:
            weighted_rk = 0
        
        if pd.notna(row['W']):
            weighted_w = (int(row['G']) / total_games) * row['W']
        else:
            weighted_w = 0
            
        if pd.notna(row['L']):
            weighted_l = (int(row['G']) / total_games) * row['L']
        else:
            weighted_l = 0
        
        return weighted_rk, weighted_w, weighted_l
    
    # Group by 'Player'
    for player, group in df.groupby('Player'):
        # Find the 'TOT' row
        tot_row = group[group['Tm'] == 'TOT']
        
        # If a 'TOT' row exists
        if not tot_row.empty:
            # Calculate the weighted Rk, W, L for other rows (not 'TOT')
            weighted_values = group[group['Tm'] != 'TOT'].apply(
                lambda x: calculate_weighted_values(x, 82), axis=1
            )
            
            # Summing up the weighted values
            weighted_rk_sum = weighted_values.apply(lambda x: x[0]).sum()
            weighted_w_sum = weighted_values.apply(lambda x: x[1]).sum()
            weighted_l_sum = weighted_values.apply(lambda x: x[2]).sum()
    
            # Set these values to the 'Rk', 'W', 'L' columns of the 'TOT' row
            df.loc[tot_row.index, 'Rk'] = weighted_rk_sum
            df.loc[tot_row.index, 'W'] = weighted_w_sum
            df.loc[tot_row.index, 'L'] = weighted_l_sum
    
    
    has_tot = df[df['Tm'] == 'TOT']['Player'].unique()
    
    # Filter the DataFrame
    # Keep only 'TOT' entries for players who have them
    # Keep all entries for players who don't have a 'TOT' entry
    df = df[(df['Player'].isin(has_tot) & (df['Tm'] == 'TOT')) | (~df['Player'].isin(has_tot))]
    
    # Showing the result
    total_stats = df.dropna()
    
    return total_stats

def clean_roy(roy):
    roy.columns = [col[1] for col in roy.columns]
    return roy

def clean_rookies(rookies):
    rookies = rookies[['Player','Age', 'G', 'MP', 'FG', 'FGA', '3P',
       '3PA', 'FT', 'FTA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
       'PTS', 'FG%', '3P%', 'FT%']]
    rookies = rookies[~(rookies['Player'].isna() | (rookies['Player'] == 'Player'))]
    rookies = rookies.fillna(.000)
    return rookies

total_stats_dfs = []
mvp_dfs = []
roy_dfs = []
rookies_dfs = []
years = range(1977, 2025)

current_directory = os.path.dirname(os.path.realpath(__file__))

data_folder_path = os.path.join(current_directory, 'data')

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

for year in years:
    if (os.path.exists(os.path.join(data_folder_path, f'stats_{year}.csv')) and os.path.exists(os.path.join(data_folder_path, f'mvp_{year}.csv')) and os.path.exists(os.path.join(data_folder_path, f'roy_{year}.csv')) and os.path.exists(os.path.join(data_folder_path, f'rookies_{year}.csv'))):
        print("csv files for:", year, "already exist")
    else:
        stats_url = f'https://www.basketball-reference.com/leagues/NBA_{year}_totals.html'
        wins_url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html'
        awards_url = f'https://www.basketball-reference.com/awards/awards_{year}.html'
        rookie_url = f'https://www.basketball-reference.com/leagues/NBA_{year}_rookies-season-stats.html'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        stats_response = requests.get(stats_url, headers=headers)
        wins_response = requests.get(wins_url, headers=headers)
        awards_response = requests.get(awards_url, headers=headers)
        rookie_response = requests.get(rookie_url, headers=headers)
        
        if stats_response.status_code == 200 and wins_response.status_code == 200 and awards_response.status_code == 200 and rookie_response.status_code == 200:
            print("Successfully fetched the pages for:", year)
            
            stats_html = BeautifulSoup(stats_response.text, "html.parser")
            wins_html = BeautifulSoup(wins_response.text, "html.parser")
            wins_html.find('tr', class_="over_header").decompose()
            awards_html = BeautifulSoup(awards_response.text, "html.parser")
            awards_html.find('tr', class_="over_header").decompose()
            rookie_html = BeautifulSoup(rookie_response.text, "html.parser")
            rookie_html.find('tr', class_="over_header").decompose()

            stats_raw = str(stats_html.find(id="totals_stats"))
            wins_raw = str(wins_html.find(id="advanced-team"))
            mvp_raw = str(awards_html.find(id="mvp"))
            roy_raw = str(awards_html.find(id="roy"))
            rookie_raw = str(rookie_html.find(id="rookies"))
            #dpoy_raw = str(awards_html.find(id="dpoy"))
            #smoy_raw = str(awards_html.find(id="smoy"))
            #mip_raw = str(awards_html.find(id="mip"))
            
            # Now, pass this file-like object to pd.read_html()
            total_stats = pd.read_html(StringIO(stats_raw))[0]
            wins = pd.read_html(StringIO(wins_raw))[0]
            mvp = pd.read_html(StringIO(mvp_raw))[0]
            roy = pd.read_html(StringIO(roy_raw))[0]
            rookies = pd.read_html(StringIO(rookie_raw))[0]
            #dpoy = pd.read_html(StringIO(dpoy_raw))[0]
            #smoy = pd.read_html(StringIO(smoy_raw))[0]
            #mip = pd.read_html(StringIO(mip_raw))[0]
        
            total_stats = clean_stats(total_stats, wins)
            roy = clean_roy(roy)
            rookies = clean_rookies(rookies)
            total_stats_dfs.append(total_stats)
            mvp_dfs.append(mvp)
            roy_dfs.append(roy)
            rookies_dfs.append(rookies)
    
            total_stats.to_csv(os.path.join(data_folder_path, f'stats_{year}.csv'), index=False)
            mvp.to_csv(fos.path.join(data_folder_path, f'mvp_{year}.csv'), index=False)
            roy.to_csv(os.path.join(data_folder_path, f'roy_{year}.csv'), index=False)
            rookies.to_csv(os.path.join(data_folder_path, f'rookies_{year}.csv'), index=False)
        elif stats_response.status_code == 429 or wins_response.status_code == 429 or awards_response.status_code == 429 or rookie_response.status_code == 429:
            retry_after = int(stats_response.headers.get('Retry-After', 30))  # Default to 30 seconds if header is missing
            print(f"Rate limit exceeded. Waiting for {retry_after} seconds.")
            time.sleep(retry_after)
        else:
            print("Failed to fetch the page, stats status code:", stats_response.status_code, "for year:", year)
            print("Failed to fetch the page, wins status code:", wins_response.status_code, "for year:", year)
            print("Failed to fetch the page, awards status code:", awards_response.status_code, "for year:", year)
            print("Failed to fetch the page, rookies status code:", rookie_response.status_code, "for year:", year)