from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        sp = float(request.form['sp'])
        strength = float(request.form['strength'])
        co2 = float(request.form['co2'])
        durability = float(request.form['durability'])

        features = np.array([[sp, strength, co2, durability]])

        prediction = model.predict(features)

        sustainability = round(prediction[0], 4)

        # Sustainability Category
        if sustainability < 0.08:
            category = "Poor"

        elif sustainability < 0.12:
            category = "Moderate"

        else:
            category = "Excellent"

        return render_template(
            'index.html',
            prediction_text=f'Sustainability Index: {sustainability}',
            category=category
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)