
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and label encoders
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Frontend HTML content (basic UI embedded in Python string)
FRONTEND_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Salary Prediction App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); max-width: 600px; margin: auto; }
        h1 { text-align: center; color: #333; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="number"], select { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; }
        button:hover { background-color: #0056b3; }
        #predictionResult { margin-top: 20px; padding: 15px; border-radius: 4px; background-color: #e9ecef; text-align: center; font-size: 1.2em; font-weight: bold; color: #333; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Salary Prediction</h1>
        <form id="predictionForm">
            <label for="age">Age:</label>
            <input type="number" id="age" name="Age" min="18" max="65" required>

            <label for="gender">Gender:</label>
            <select id="gender" name="Gender" required>
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>

            <label for="educationLevel">Education Level:</label>
            <select id="educationLevel" name="Education Level" required>
                <option value="">Select Education Level</option>
                <option value="High School">High School</option>
                <option value="Bachelor's Degree">Bachelor's Degree</option>
                <option value="Master's Degree">Master's Degree</option>
                <option value="PhD">PhD</option>
                <option value="Associate's Degree">Associate's Degree</option>
                <option value="Some College">Some College</option>
            </select>

            <label for="jobTitle">Job Title:</label>
            <select id="jobTitle" name="Job Title" required>
                <option value="">Select Job Title</option>
                <option value="Software Engineer">Software Engineer</option>
                <option value="Data Analyst">Data Analyst</option>
                <option value="Senior Manager">Senior Manager</option>
                <option value="Sales Associate">Sales Associate</option>
                <option value="Director">Director</option>
                <option value="Marketing Manager">Marketing Manager</option>
                <option value="Financial Manager">Financial Manager</option>
                <option value="HR Manager">HR Manager</option>
                <option value="Project Manager">Project Manager</option>
                <option value="Data Scientist">Data Scientist</option>
                <option value="Business Analyst">Business Analyst</option>
                <option value="IT Support">IT Support</option>
            </select>

            <label for="yearsOfExperience">Years of Experience:</label>
            <input type="number" id="yearsOfExperience" name="Years of Experience" min="0" max="40" required>

            <button type="submit">Predict Salary</button>
        </form>
        <div id="predictionResult"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(event) {
            event.preventDefault(); // Prevent default form submission

            const form = event.target;
            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => { data[key] = value; });

            const resultDiv = document.getElementById('predictionResult');
            resultDiv.innerHTML = 'Predicting...';
            resultDiv.classList.remove('error');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Prediction failed');
                }

                const result = await response.json();
                resultDiv.innerHTML = `Predicted Salary: $${result.predicted_salary.toLocaleString()}`;
            } catch (error) {
                resultDiv.innerHTML = `Error: ${error.message}`;
                resultDiv.classList.add('error');
                console.error('Prediction error:', error);
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(FRONTEND_HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Prepare input data
        input_df = pd.DataFrame([data])

        # Apply label encoding to categorical features
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                # Handle unseen labels by ensuring the value exists in the encoder's classes
                if input_df[col][0] not in encoder.classes_:
                    return jsonify({'error': f"Unknown value '{input_df[col][0]}' for feature '{col}'"}), 400
                input_df[col] = encoder.transform(input_df[col])

        # Ensure numeric columns are correctly typed
        input_df['Age'] = pd.to_numeric(input_df['Age'])
        input_df['Years of Experience'] = pd.to_numeric(input_df['Years of Experience'])

        # Ensure the order of columns is the same as the training data
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
