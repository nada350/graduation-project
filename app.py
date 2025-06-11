import json
import pandas as pd
import os

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
from datetime import datetime  



@register_keras_serializable()
def f1_score(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())

    return 2 * (precision * recall) / (precision + recall + K.epsilon())


  

app = Flask(__name__)
CORS(app)

model = load_model('C:/Users/LENOVO/Desktop/vs/fraud_model_checkpoint.keras', custom_objects={'f1_score': f1_score})

df = pd.read_csv("train_data_preprocessed.csv")

@app.route('/')
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route('/process_payment', methods=['POST'])
def process_payment():
    try:
        data = request.get_json()
        name_orig = data.get("nameOrig")

        if not name_orig:
            return jsonify({"error": "nameOrig is required"}), 400

        

      
        if df["nameOrig"].dtype == 'float64':
            name_orig = float(name_orig)
        elif df["nameOrig"].dtype == 'int64':
            name_orig = int(name_orig)
        else:
            name_orig = str(name_orig)

       
        row = df[df["nameOrig"] == name_orig]
       

        if row.empty:
            return jsonify({"error": "User not found in dataset"}), 404

      
        features = np.array([[float(row["step"]),
                              float(row["type"]),
                              float(row["amount"]),
                              float(row["nameOrig"]),
                              float(row["oldbalanceOrg"]),
                              float(row["newbalanceOrig"]),
                              float(row["nameDest"]),
                              float(row["oldbalanceDest"]),
                              float(row["newbalanceDest"])]])
        
      
        prediction = model.predict(features)
        is_fraud = int(prediction[0][0] > 0.5)
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": float(row["step"]),
            "type": float(row["type"]),
            "nameOrig": float(row["nameOrig"]),
            "nameDest": float(row["nameDest"]),
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
            "isFraud": is_fraud
        }

    
        history_path = "payment_history.json"
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append(history_entry)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        if is_fraud:
            return jsonify({"status": "failed", "message": "Fraud detected ❌"})
        else:
            return jsonify({"status": "success", "message": "Payment successful ✅"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
