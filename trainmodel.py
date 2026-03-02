# train_model.py

"""
This script trains a Machine Learning model
and saves it as model.pkl
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load Dataset
# -----------------------------

df = pd.read_csv("dataset.csv")

# -----------------------------
# 2. Basic Preprocessing
# -----------------------------

# Drop rows where target is missing
df = df.dropna(subset=["Salary"])

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------
# 3. Define Features and Target
# -----------------------------

X = df.drop("Salary", axis=1)
y = df["Salary"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train Model
# -----------------------------

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

print("R2 Score:", r2_score(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions, squared=False))

# -----------------------------
# 7. Save Model
# -----------------------------

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved as model.pkl")
