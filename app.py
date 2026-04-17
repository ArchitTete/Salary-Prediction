
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and label encoders
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Prepare input data
        input_df = pd.DataFrame([data])

        # Apply label encoding to categorical features
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                # Handle unseen labels by mapping them to an existing one or a default
                # For simplicity, we'll assume new data will have labels seen during training
                # A more robust solution might include handling unknown categories.
                input_df[col] = encoder.transform(input_df[col])

        # Ensure the order of columns is the same as the training data
        # x is the DataFrame used for training (features only)
        # It's important that `x` is defined and accessible or its columns saved.
        # Assuming `x`'s columns represent the order needed.
        # If x.columns is not available, you would need to save feature names during training.
        feature_order = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'] # Hardcoding for demonstration, in production load this.
        input_df = input_df[feature_order]

        prediction = model.predict(input_df)

        return jsonify({'predicted_salary': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local development, you can run:
    # app.run(debug=True)
    # For deployment, you might need a production-ready WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5000)
