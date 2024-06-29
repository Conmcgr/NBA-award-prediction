import joblib
import pandas as pd

# Load the trained models
mvp_model = joblib.load('models/mvp_model.pkl')
roy_model = joblib.load('models/roy_model.pkl')

def load_model(award_type):
    if award_type == 'MVP':
        return mvp_model
    elif award_type == 'ROY':
        return roy_model
    else:
        raise ValueError("Invalid award type. Choose 'MVP' or 'ROY'.")

def preprocess_data(data):
    # Keep player names
    players = data['Player']
    data = data.drop(['Player', 'Tm'], axis=1, errors='ignore')
    
    # Handle missing values
    data = data.fillna(0)
    
    # Encode categorical variables if any (assuming 'Pos' is a categorical variable)
    if 'Pos' in data.columns:
        data = pd.get_dummies(data, columns=['Pos'], drop_first=True)
    
    return data, players

def predict_award(data, award_type):
    model = load_model(award_type)
    predictions = model.predict(data)
    return predictions
