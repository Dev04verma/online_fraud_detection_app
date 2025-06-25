from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html', result=None, sample=None)

# ✅ Sample loader route
@app.route('/sample')
def load_sample():
    sample = {
        'amount': 10000,
        'oldbalanceOrg': 0,
        'newbalanceOrig': 0,
        'oldbalanceDest': 0,
        'newbalanceDest': 10000,
        'type': 'TRANSFER'
    }
    return render_template('index.html', result=None, sample=sample)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        trans_type = request.form['type']

        # Encode type
        type_CASH_IN = 0
        type_CASH_OUT = 0
        type_TRANSFER = 0

        if trans_type == "CASH_IN":
            type_CASH_IN = 1
        elif trans_type == "CASH_OUT":
            type_CASH_OUT = 1
        elif trans_type == "TRANSFER":
            type_TRANSFER = 1

        # Form input array
        input_data = np.array([[amount, oldbalanceOrg, newbalanceOrig,
                                oldbalanceDest, newbalanceDest,
                                type_CASH_IN, type_CASH_OUT, type_TRANSFER]])

        # Predict
        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legitimate Transaction"
        return render_template('index.html', result=result, sample=None)

if __name__ == '__main__':
    app.run(debug=True)
