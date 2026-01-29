import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. Define All 20 Attributes ---
# Numeric: Variables with a range of numbers
num_cols = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", 
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
]

# Categorical: Yes/No or Normal/Abnormal
cat_cols = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", 
    "cad", "appet", "pe", "ane"
]

# --- 2. Load and Initial Prep ---
f = pd.read_csv('data/kidney disease dataset.csv')

# Professional fix: Ensure mixed-type columns are numeric before the pipeline
for col in ['pcv', 'wc', 'rc']:
    f[col] = pd.to_numeric(f[col], errors='coerce')

# Map target variable
f['classification'] = f['classification'].apply(lambda x: 1 if 'ckd' in str(x).lower() else 0)

# --- 3. Build the Pipeline ---
# This ensures data cleaning is handled professionally
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Final model with 100 trees for stability
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- 4. Train and Serialize ---
X = f[num_cols + cat_cols]
y = f['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

# Save the entire pipeline (includes all 20 attributes logic)
with open("backend/rf_pipeline.pkl", "wb") as file:
    pickle.dump(model_pipeline, file)

print("SUCCESS: Pipeline saved with all 20 attributes.")