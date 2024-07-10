from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models
print("Loading models...")
mvp_base_model = joblib.load('models/mvp_base_model.pkl')
roy_base_model = joblib.load('models/roy_base_model.pkl')
print("Models loaded successfully.")

def load_model(award_type):
    if award_type == 'MVP':
        return mvp_base_model
    elif award_type == 'ROY':
        return roy_base_model
    else:
        raise ValueError("Invalid award type. Choose 'MVP' or 'ROY'.")

def preprocess_data(data, required_columns):
    players = data['Player']
    data = data.drop(['Player', 'Tm'], axis=1, errors='ignore')
    data = data.fillna(0)
    if 'Pos' in data.columns:
        data = pd.get_dummies(data, columns=['Pos'], drop_first=True)
    for col in required_columns:
        if col not in data.columns:
            data[col] = 0
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

        data = pd.read_csv(file)
        model = load_model(award_type)
        if award_type == 'MVP':
            required_columns = mvp_base_model.feature_names_in_
        else:
            required_columns = roy_base_model.feature_names_in_

        processed_data, players = preprocess_data(data, required_columns)
        predictions = model.predict(processed_data)
        
        results = pd.DataFrame({
            'Player': players,
            'Predicted Vote Share': predictions
        })
        
        top_10 = results.nlargest(10, 'Predicted Vote Share')
        return render_template('results.html', tables=[top_10.to_html(classes='data', header="true")])

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(debug=True)
