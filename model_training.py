import pandas as pd
import numpy as np
import random
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from data_collection_cleaning import data_folder_path

start_year = 1977
end_year = 2024
years = range(start_year, end_year+1)
mvps = {}
roys = {}
stats = {}
rookies = {}

for year in years:
    mvps[year] = pd.read_csv(os.path.join(data_folder_path, f'mvp_{year}.csv'))
    mvps[year]['Year'] = year
    roys[year] = pd.read_csv(os.path.join(data_folder_path, f'roy_{year}.csv'))
    roys[year]['Year'] = year
    stats[year] = pd.read_csv(os.path.join(data_folder_path, f'stats_{year}.csv'))
    stats[year]['Year'] = year
    rookies[year] = pd.read_csv(os.path.join(data_folder_path, f'rookies_{year}.csv'))
    rookies[year] = pd.merge(rookies[year], stats[year][['Player','Tm', 'Pos', 'Rk', 'W', 'L']], on='Player', how='left')
    rookies[year]['Pos'] = rookies[year]['Pos'].fillna('Unkown')
    rookies[year]['Year'] = year

def merge_stats_share(stats, award, new_col_name):
    #Remove asterix from names
    stats['Player'] = stats['Player'].str.replace('*', '', regex=False)
    award['Player'] = award['Player'].str.replace('*', '', regex=False)

    #Normalize voting shares so they sum to 100
    total_share = sum(award['Share'])
    award['Share'] = award['Share']*100/total_share
    
    merge = pd.merge(stats, award[['Player', 'Share']], on='Player', how='left')
    merge = merge.fillna(0.000)
    merge = merge.rename(columns={'Share': new_col_name})
    return merge

mvp_stats_merges = {}
roy_rookies_merges = {}

for year in years:
    mvp_stats_merges[year] = merge_stats_share(stats[year],mvps[year],'MVP Vote Share')
    roy_rookies_merges[year] = merge_stats_share(rookies[year],roys[year],'ROY Vote Share')
