import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# Load Dataset
# =========================

df = pd.read_excel('sustainability.xlsx')

print(df.head())

# =========================
# Convert Strength Ranges
# =========================

def convert_strength(value):

    if isinstance(value, str):

        if '–' in value:
            parts = value.split('–')
            return (float(parts[0]) + float(parts[1])) / 2

        elif '-' in value:
            parts = value.split('-')
            return (float(parts[0]) + float(parts[1])) / 2

    return float(value)

df['Strength (MPa)'] = df['Strength (MPa)'].apply(convert_strength)

# =========================
# Features
# =========================

X = df[['SP (%)', 'Strength (MPa)', 'CO₂ (kg/m³)', 'Durability']]

# =========================
# Target
# =========================

y = df['Sustainability']

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
# Train
# =========================

model.fit(X_train, y_train)

# =========================
# Predict
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

pickle.dump(model, open('model.pkl', 'wb'))

print("\nModel saved as model.pkl")