# Training_and_saving_GBR_models

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ----------------------------
# Load dataset Data_set (previous) and Data_set (updated) after reshuffling
# ----------------------------
data_path = 'Data_set (reshuffled).csv'
train_data = pd.read_csv(data_path)

# ----------------------------
# Feature selection per target
# ----------------------------

# Features for Yield Strength (YS)
selected_for_YS = [
    'Si', 'Fe', 'Cu', 'Mg', 'Zn', 'Ti', 'Zr', 'Sc', 'Mn', 'Al',
    'SHT_Temperature', 'SHT_Time', 'Aging_temperature', 'Aging_Time',
    'Process', 'Valence_electrons', 'Group', 'Heat_of_fusion',
    'Density', 'Atomic_radius', 'Shear_modulus', 'Poisson_ratio',
    'Mendeleev_number', 'Space_group', 'First_ionization_energy'
]

# Features for Elongation (El)
selected_for_El = [
    'Si', 'Fe', 'Cu', 'Mg', 'Zn', 'Ti', 'Zr', 'Sc', 'Mn', 'Al',
    'SHT_Temperature', 'SHT_Time', 'Aging_temperature', 'Aging_Time',
    'Process', 'Melting_temperature', 'Atomic_number',
    'Bulk_modulus', 'Poisson_ratio', 'Mendeleev_number',
    'Electronegativity_allred', 'Space_group', 'First_ionization_energy'
]

# ----------------------------
# Define grid search parameters
# ----------------------------
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}

# ----------------------------
# Train and save models
# ----------------------------
targets = ['YS', 'El']
feature_sets = [selected_for_YS, selected_for_El]

for target, selected_features in zip(targets, feature_sets):
    print(f"\nTraining and tuning model for {target}...")

    # Define features and labels
    X = train_data[selected_features]
    y = train_data[target]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model + hyperparameter tuning
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_scaled, y)

    # Retrieve best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters for {target}: {grid_search.best_params_}")

    # Save model and scaler
    joblib.dump(best_model, f"{target}_model.joblib")
    joblib.dump(scaler, f"{target}_scaler.joblib")

    print(f"Saved best model and scaler for {target}.")