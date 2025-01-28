from flask_cors import CORS
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import os
import json

# Flask App Setup
app = Flask(__name__)

# Step 1: Generate Synthetic Data
def generate_synthetic_data():
    np.random.seed(42)
    
    # Simulate inputs
    age = np.random.randint(10, 60, 1000)
    hours_played = np.random.randint(1, 20, 1000)
    intensity_level = np.random.choice(["low", "medium", "high"], 1000)
    sport = np.random.choice(["soccer", "basketball", "tennis", "swimming"], 1000)
    
    # Map intensity levels to numerical values
    intensity_map = {"low": 0, "medium": 1, "high": 2}
    intensity_numeric = [intensity_map[level] for level in intensity_level]
    
    # Injury risk logic (simplified)
    risk = []
    for i in range(1000):
        if hours_played[i] > 5 and intensity_numeric[i] == 2:
            risk.append(1)  # High risk
        elif hours_played[i] > 7 and intensity_numeric[i] == 1:
            risk.append(1)  # Medium risk
        elif age[i] < 15 or age[i] > 50:
            risk.append(1)  # High risk due to age
        else:
            risk.append(0)  # Low risk

    # Create DataFrame
    data = pd.DataFrame({
        "age": age,
        "hours_played": hours_played,
        "intensity": intensity_numeric,
        "sport": sport,
        "risk": risk
    })
    return data

# Step 2: Train Models
def train_models():
    data = generate_synthetic_data()
    X = data[["age", "hours_played", "intensity"]]
    y = data["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # K-NN Classifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    return lr_model, knn_model, X_test, y_test

# Train models on startup
lr_model, knn_model, X_test, y_test = train_models()

# Step 3: API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = data["age"]
        hours_played = data["hours_played"]
        intensity = data["intensity"]  # Numeric (0: low, 1: medium, 2: high)

        input_data = np.array([[age, hours_played, intensity]])
        prediction = lr_model.predict(input_data)[0]
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({"risk_level": risk_level})
    except Exception as e:
        return jsonify({"error": str(e)})

# Step 4: API Endpoint for Visualizations
@app.route('/visualize', methods=['GET'])
def visualize():
    try:
        # Confusion Matrix
        y_pred = lr_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Save Confusion Matrix Plot
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')

        # ROC Curve
        y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')

        return jsonify({"message": "Visualizations generated", "roc_auc": roc_auc})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
