from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# =========================
# Load Model
# =========================

model = pickle.load(open('carbon_model.pkl', 'rb'))

# =========================
# Home
# =========================

@app.route('/')
def home():
    return render_template('index.html')

# =========================
# Predict
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        cement = float(request.form['cement'])

        water = float(request.form['water'])

        fa = float(request.form['fa'])

        ggbfs = float(request.form['ggbfs'])

        sf = float(request.form['sf'])

        sp = float(request.form['sp'])

        ca = float(request.form['ca'])

        features = np.array([
            [
                cement,
                water,
                fa,
                ggbfs,
                sf,
                sp,
                ca
            ]
        ])

        prediction = model.predict(features)

        carbon = round(prediction[0], 2)

        # Category
        if carbon < 300:
            category = "Low Carbon Mix"

        elif carbon < 400:
            category = "Moderate Carbon Mix"

        else:
            category = "High Carbon Mix"

        return render_template(
            'index.html',
            prediction_text=f'Predicted Carbon Footprint: {carbon} kg/m³',
            category=category
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

# =========================
# Run
# =========================

if __name__ == '__main__':
    app.run(debug=True)
