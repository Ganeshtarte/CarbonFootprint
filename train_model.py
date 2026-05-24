import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
        'GGBS (%)',
        'Silica Fume (%)',
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
# Random Forest Model
# =========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# =========================
# Train
# =========================

model.fit(X_train, y_train)

# =========================
# Save Model
# =========================

pickle.dump(model, open('carbon_model.pkl', 'wb'))

print("Carbon Footprint Model Saved Successfully")
