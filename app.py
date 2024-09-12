from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load the models
mvp_model = joblib.load('models/mvp_base_model.pkl')
roy_model = joblib.load('models/roy_base_model.pkl')


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    award_type = request.form['award_type']
    data_source = request.form['data_source']
    
    if data_source == 'upload':
        file = request.files['file']
        df = pd.read_csv(file)
        year = None
    else:
        year = int(request.form['year'])
        df = pd.read_csv(f'data/stats_{year}.csv')
    
    if award_type == 'MVP':
        X = preprocess_data(df, year=year, model=mvp_model)
        predictions = mvp_model.predict(X)
        df['Prediction'] = predictions
    else:
        # Use rookie stats for ROY predictions
        df = pd.read_csv(f'data/rookies_{year}.csv')
        
        # Create per-48 minute stats for rookies
        for col in df.columns:
            if col not in ['Player', 'Age', 'G', 'MP', 'FG%', '3P%', 'FT%']:
                df[f'{col}P48'] = df[col] / df['MP'] * 48
        
        # Ensure all expected features are present
        for feature in roy_model.feature_names_in_:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[roy_model.feature_names_in_]
        predictions = roy_model.predict(X)
        df['Prediction'] = predictions

    # Emphasize top performers
    df['Prediction'] = np.power(df['Prediction'], 3)

    # Get top 5
    top_5 = df.nlargest(5, 'Prediction')

    # Normalize to 100%
    top_5['Prediction'] = (top_5['Prediction'] / top_5['Prediction'].sum()) * 100

    simplified_results = top_5[['Player', 'Prediction']].rename(columns={'Prediction': 'Predicted Vote Share'})
    full_results = top_5

    return render_template('results.html', 
                        simplified_table=simplified_results.to_html(classes='data', index=False),
                        full_table=full_results.to_html(classes='data', index=False),
                        titles=simplified_results.columns.values)

def preprocess_data(df, year=None, model=mvp_model):
    expected_features = model.feature_names_in_

    # Add 'Year' column if not present, using the provided year or the first year in the data
    if 'Year' not in df.columns:
        if year:
            df['Year'] = year
        else:
            df['Year'] = df['Year'].iloc[0] if 'Year' in df.columns else datetime.datetime.now().year

    # Create per-48 minute stats
    for col in df.columns:
        if col not in ['Player', 'Pos', 'Tm', 'Year']:
            df[f'{col}P48'] = df[col] / df['MP'] * 48

    # One-hot encode the 'Pos' column
    pos_dummies = pd.get_dummies(df['Pos'], prefix='Pos')
    df = pd.concat([df, pos_dummies], axis=1)

    # Create a DataFrame with all expected features, filled with zeros
    df_processed = pd.DataFrame(0, index=df.index, columns=expected_features)

    # Fill in the values for the features we have
    for col in expected_features:
        if col in df.columns:
            df_processed[col] = df[col]

    # Handle missing values
    df_processed = df_processed.fillna(0)

    # Normalize the data
    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            # Convert boolean columns to int (0 or 1)
            df_processed[col] = df_processed[col].astype(int)
        if df_processed[col].max() != 0:
            df_processed[col] = df_processed[col] / df_processed[col].max()

        return df_processed

if __name__ == '__main__':
    app.run(debug=True)
