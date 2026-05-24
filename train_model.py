import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# Load Dataset
# =========================

df = pd.read_excel('sustainability.xlsx')

# =========================
# Features
# =========================

X = df[
    [
        'Cement (kg/m³)',
        'Water (kg/m³)',
        'FA (kg/m³)',
        'GGBFS (kg/m³)',
        'SF (kg/m³)',
        'SP (%)',
        'CA (kg/m³)'
    ]
]

# =========================
# Target
# =========================

y = df['CO₂ (kg/m³)']

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Model
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# =========================
# Train Model
# =========================

model.fit(X_train, y_train)

# =========================
# Prediction
# =========================

y_pred = model.predict(X_test)

# =========================
# Accuracy
# =========================

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("R2 Score :", r2)
print("MAE      :", mae)

# =========================
# Save Model
# =========================

pickle.dump(model, open('carbon_model.pkl', 'wb'))

print("\nCarbon Model saved successfully")
