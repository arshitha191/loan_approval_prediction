from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import webbrowser
import threading

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and preprocess input
    features = [
        data['Gender'], data['Married'], data['Dependents'], data['Education'],
        data['Self_Employed'], data['ApplicantIncome'], data['CoapplicantIncome'],
        data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'],
        data['Property_Area']
    ]

    # Convert categorical to numerical (must match model training)
    gender = 1 if features[0] == 'Male' else 0
    married = 1 if features[1] == 'Yes' else 0
    dependents = 3 if features[2] == '3+' else int(features[2])
    education = 0 if features[3] == 'Graduate' else 1
    self_employed = 1 if features[4] == 'Yes' else 0
    property_area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[features[10]]

    # Final input to model
    final_features = np.array([[ 
        gender, married, dependents, education, self_employed, 
        float(features[5]), float(features[6]), float(features[7]), 
        float(features[8]), float(features[9]), property_area 
    ]])

    prediction = model.predict(final_features)
    result = 'Approved' if prediction[0] == 1 else 'Rejected'

    return jsonify({'prediction': result})

def open_browser():
    # Wait for the server to start, then open in the default browser
    import time
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5001")

if __name__ == '__main__':
    # Start the browser in a separate thread
    threading.Thread(target=open_browser).start()
    app.run(debug=True, port=5001)
