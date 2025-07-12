# predict_tensile_properties_of_new_Al_Fe_based_alloys.py

import pandas as pd
import joblib

# Define selected features for each target
selected_for_YS = ['Si', 'Fe', 'Cu', 'Mg', 'Zn', 'Ti', 'Zr', 'Sc', 'Mn', 'Al',
                   'SHT_Temperature', 'SHT_Time', 'Aging_temperature', 'Aging_Time',
                   'Process', 'Valence_electrons', 'Group', 'Heat_of_fusion',
                   'Density', 'Atomic_radius', 'Shear_modulus', 'Poisson_ratio',
                   'Mendeleev_number', 'Space_group', 'First_ionization_energy']

selected_for_El = ['Si', 'Fe', 'Cu', 'Mg', 'Zn', 'Ti', 'Zr', 'Sc', 'Mn', 'Al',
                   'SHT_Temperature', 'SHT_Time', 'Aging_temperature', 'Aging_Time',
                   'Process', 'Melting_temperature', 'Atomic_number',
                   'Bulk_modulus', 'Poisson_ratio', 'Mendeleev_number',
                   'Electronegativity_allred', 'Space_group', 'First_ionization_energy']

# Load alloy_with_weighted_elemental_properties which contain composition, processing conditions, and elemental properties

new_alloys = pd.read_csv('alloy_with_weighted_elemental_properties.csv')

# DataFrame to store predictions
predictions_df = pd.DataFrame()

# Predict for each target using saved models
for target, selected_features in zip(['YS', 'El'], [selected_for_YS, selected_for_El]):
    print(f"\\nPredicting {target} for new alloys...")

    # Load saved model and scaler
    model = joblib.load(f"{target}_model(n).joblib")
    scaler = joblib.load(f"{target}_scaler(n).joblib")

    # Prepare features
    X_new = new_alloys.copy()

    # Ensure all selected features are present
    for col in selected_features:
        if col not in X_new.columns:
            X_new[col] = 0  # or np.nan, depending on use case

    # Reorder columns to match training
    X_new = X_new[selected_features]

    # Scale using training scaler
    X_new_scaled = scaler.transform(X_new)

    # Predict
    predictions_df[target] = model.predict(X_new_scaled)

# Add predictions to original new_alloys DataFrame
new_alloys['Predicted_YS'] = predictions_df['YS']
new_alloys['Predicted_El'] = predictions_df['El']

# Save predictions to CSV
output_file = 'predicted_YS_El_of_new Al-Fe alloys.csv'
new_alloys.to_csv(output_file, index=False)

print(f"\\nPredictions saved to {output_file}")
'''

with open("predict_new_alloys.py", "w") as f:
    f.write(code)

print("✅ Script saved as 'predict_new_alloys.py'")

