import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
import joblib
import heapq
from data_collection_cleaning import end_year, start_year, years, all_mvp_data_per_48, all_roy_data_per_48, mvp_48_with_categorical, roy_48_with_categorical


# Split training and testing data (keep each year's data together)
test_percentage = 0.2
num_years_test = int(test_percentage * (end_year - start_year))
test_years = random.sample(years, num_years_test)

mvp_train_per_48 = all_mvp_data_per_48[all_mvp_data_per_48['Year'].isin(test_years) == False]
mvp_test_per_48 = all_mvp_data_per_48[all_mvp_data_per_48['Year'].isin(test_years)]

roy_train_per_48 = all_roy_data_per_48[all_roy_data_per_48['Year'].isin(test_years) == False]
roy_test_per_48 = all_roy_data_per_48[all_roy_data_per_48['Year'].isin(test_years)]

# Create Features --> X = features used to predict, Y = feature to predict
mvp_x_train_per_48 = mvp_train_per_48.drop(['MVP Vote Share'], axis=1)
mvp_x_test_per_48 = mvp_test_per_48.drop(['MVP Vote Share'], axis=1)

mvp_y_train_per_48 = mvp_train_per_48['MVP Vote Share']
mvp_y_test_per_48 = mvp_test_per_48['MVP Vote Share']

roy_x_train_per_48 = roy_train_per_48.drop(['ROY Vote Share'], axis=1)
roy_x_test_per_48 = roy_test_per_48.drop(['ROY Vote Share'], axis=1)

roy_y_train_per_48 = roy_train_per_48['ROY Vote Share']
roy_y_test_per_48 = roy_test_per_48['ROY Vote Share']

# Function to create prediction model
def create_rfr_preds(x_train, y_train, x_test, y_test, num_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    # Initialize the model
    model = RandomForestRegressor(n_estimators=num_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)

    # Fit the model
    model.fit(x_train, y_train)

    # Predict and evaluate
    preds = model.predict(x_test)
    
    # Cut off low values and normalize
    top_8 = heapq.nlargest(8, preds)[-1]
    significant_vote_threshold = top_8
    preds[preds < significant_vote_threshold] = 0
    preds = (preds / sum(preds)) * 100
    
    mse = mean_squared_error(y_test, preds)

    return [model, preds, mse]

# MVP Model
mvp_per_48_rfr = create_rfr_preds(mvp_x_train_per_48, mvp_y_train_per_48, mvp_x_test_per_48, mvp_y_test_per_48, num_estimators=50, max_depth=10, min_samples_leaf=1, min_samples_split=2)
mvp_per_48_model = mvp_per_48_rfr[0]
mvp_per_48_preds = mvp_per_48_rfr[1]
mvp_per_48_mse = mvp_per_48_rfr[2]
                         
print(f'Mean Squared Error: {mvp_per_48_mse}')

calibrated_mvp_model = IsotonicRegression(out_of_bounds='clip')
calibrated_mvp_model.fit(mvp_per_48_preds, mvp_y_test_per_48)

# Apply isotonic regression to adjust the predictions
calibrated_mvp_preds_iso = calibrated_mvp_model.predict(mvp_per_48_preds)

# Evaluate the calibrated predictions
calibrated_mvp_mse_iso = mean_squared_error(mvp_y_test_per_48, calibrated_mvp_preds_iso)
print(f'MVP Calibrated Mean Squared Error with Isotonic Regression: {calibrated_mvp_mse_iso}')

# Per 48 ROY Model
roy_per_48_rfr = create_rfr_preds(roy_x_train_per_48, roy_y_train_per_48, roy_x_test_per_48, roy_y_test_per_48, num_estimators=74, max_depth=10, min_samples_split=2, min_samples_leaf=30)
roy_per_48_model = roy_per_48_rfr[0]
roy_per_48_preds = roy_per_48_rfr[1]
roy_per_48_mse = roy_per_48_rfr[2]
                         
print(f'Mean Squared Error: {roy_per_48_mse}')

calibrated_roy_model = IsotonicRegression(out_of_bounds='clip')
calibrated_roy_model.fit(roy_per_48_preds, roy_y_test_per_48)

# Apply isotonic regression to adjust the predictions
calibrated_roy_preds_iso = calibrated_roy_model.predict(roy_per_48_preds)

