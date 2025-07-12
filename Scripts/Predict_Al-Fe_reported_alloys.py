# predict_Al-Fe reported alloys.py

import pandas as pd
import joblib
import numpy as np

# ----------------------------
# Feature sets used during training
# ----------------------------
selected_for_YS = [
    'Si', 'Fe', 'Cu', 'Mg', 'Zn', 'Ti', 'Zr', 'Sc', 'Mn', 'Al',
    'SHT_Temperature', 'SHT_Time', 'Aging_temperature', 'Aging_Time',
    'Process', 'Valence_electrons', 'Group', 'Heat_of_fusion',
    'Density', 'Atomic_radius', 'Shear_modulus', 'Poisson_ratio',
    'Mendeleev_number', 'Space_group', 'First_ionization_energy'
]

selected_for_El = [
    'Si', 'Fe', 'Cu', 'Mg', 'Zn', 'Ti', 'Zr', 'Sc', 'Mn', 'Al',
    'SHT_Temperature', 'SHT_Time', 'Aging_temperature', 'Aging_Time',
    'Process', 'Melting_temperature', 'Atomic_number',
    'Bulk_modulus', 'Poisson_ratio', 'Mendeleev_number',
    'Electronegativity_allred', 'Space_group', 'First_ionization_energy'
]

# ----------------------------
# Load experimental alloy data
# ----------------------------
new_alloys = pd.read_csv('Al_Fe_alloys_literature_data.csv')

# ----------------------------
# DataFrame to store predictions
# ----------------------------
predictions_df = pd.DataFrame()

# ----------------------------
# Run predictions for each target
# ----------------------------
for target, selected_features in zip(['YS', 'El'], [selected_for_YS, selected_for_El]):
    print(f"\nPredicting {target} for new alloys...")

    # Load model and scaler
    model = joblib.load(f"{target}_model.joblib")
    scaler = joblib.load(f"{target}_scaler.joblib")

    # Copy and validate feature columns
    X_new = new_alloys.copy()
    for col in selected_features:
        if col not in X_new.columns:
            print(f"Warning: {col} missing from input. Filling with 0.")
            X_new[col] = 0  # Fallback if column missing

    # Reorder columns to match training
    X_new = X_new[selected_features]

    # Scale features
    X_scaled = scaler.transform(X_new)

    # Predict
    predictions_df[target] = model.predict(X_scaled)

# ----------------------------
# Combine predictions with original data
# ----------------------------
new_alloys['Predicted_YS'] = predictions_df['YS']
new_alloys['Predicted_El'] = predictions_df['El']

# Save predictions
new_alloys.to_csv('Predicted_Experimental_Alloys.csv', index=False)

# Print results
print("\nPredictions for new alloys:")
print(new_alloys[['Predicted_YS', 'Predicted_El']])