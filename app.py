from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models
print("Loading models...")
mvp_base_model = joblib.load('models/mvp_base_model.pkl')
roy_base_model = joblib.load('models/roy_base_model.pkl')
print("Models loaded successfully.")

def load_models(award_type):
    if award_type == 'MVP':
        return mvp_base_model
    elif award_type == 'ROY':
        return roy_base_model
    else:
        raise ValueError("Invalid award type. Choose 'MVP' or 'ROY'.")

def preprocess_data(data, required_columns):
    # Keep player names
    players = data['Player']
    data = data.drop(['Player', 'Tm'], axis=1, errors='ignore')
    
    # Handle missing values
    data = data.fillna(0)
    
    # Encode categorical variables if any (assuming 'Pos' is a categorical variable)
    if 'Pos' in data.columns:
        data = pd.get_dummies(data, columns=['Pos'], drop_first=True)
    
    # Ensure all required columns are present
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with default value 0
    
    # Ensure the columns are in the correct order
    data = data[required_columns]
    
    return data, players

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        award_type = request.form['award_type']
        file = request.files['file']
        if not file:
            return "No file uploaded", 400
        
        print("File uploaded successfully.")
        data = pd.read_csv(file)
        print("CSV file read successfully.")
        
        base_model = load_models(award_type)
        print(f"Model loaded for {award_type}.")
        
        # Get the required columns from the training data
        if award_type == 'MVP':
            required_columns = mvp_base_model.feature_names_in_
        else:
            required_columns = roy_base_model.feature_names_in_
        
        processed_data, players = preprocess_data(data, required_columns)
        print("Data preprocessed successfully.")
        
        print(f"Processed Data: {processed_data.head()}")
        
        predictions = base_model.predict(processed_data)
        print(f"Model Predictions: {predictions[:10]}")
        
        results = pd.DataFrame({
            'Player': players,
            'Prediction': predictions
        })
        
        # Get top 5 predictions
        top_5 = results.nlargest(5, 'Prediction')
        print(f"Top 5 Predictions: {top_5}")
        
        output_path = 'top_5_predictions.csv'
        top_5.to_csv(output_path, index=False)
        print("Top 5 predictions saved successfully.")
        
        return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(debug=True)
