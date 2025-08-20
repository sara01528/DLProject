from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

app = Flask(__name__)

def train_model():
    # Load data
    data = pd.read_csv("TrafficTwoMonth.csv")
    
    # Convert time to hour and day of the week with explicit format
    data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour
    data['Day'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce').dt.dayofweek
    
    # Drop rows with invalid date/time
    data.dropna(subset=['Time', 'Day'], inplace=True)
    
    # Select relevant columns
    data = data[['Time', 'Day', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Traffic Situation']]
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X.fillna(X.mean(), inplace=True)
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "label_encoder.pkl")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, "scaler.pkl")
    
    # Define classifiers
    cl1 = LogisticRegression()
    cl2 = RandomForestClassifier()
    cl3 = SVC(probability=True)
    cl4 = XGBClassifier()
    cl5 = AdaBoostClassifier()
    
    voting_cl = VotingClassifier(estimators=[('lr', cl1), ('rf', cl2), ('svc', cl3), ('xgb', cl4), ('ada', cl5)], voting='hard')
    
    # Train model
    voting_cl.fit(X_scaled, y_encoded)
    joblib.dump(voting_cl, "model.pkl")

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except:
    train_model()
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_names = ['Time', 'Day', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
        input_data = [float(x) for x in request.form["features"].split(",")]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        # Decode the prediction
        prediction_label = label_encoder.inverse_transform([prediction[0]])[0].lower()
        
        # Improved traffic mapping
        traffic_mapping = {
            "heavy": "Heavy Traffic",
            "high": "Heavy Traffic",
            "moderate": "Moderate Traffic",
            "normal": "Moderate Traffic",
            "low": "Low Traffic"
        }
        
        # Get the mapped value or default to the original label
        traffic_level = traffic_mapping.get(prediction_label, prediction_label)
        
        return jsonify({
            "prediction": prediction_label,  # original prediction
            "traffic_level": traffic_level  # mapped value
        })
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True)