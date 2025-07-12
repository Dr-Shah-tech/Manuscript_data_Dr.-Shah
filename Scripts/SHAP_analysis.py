# SHAP_Analysis.py

import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from matplotlib import rcParams

# Set Times New Roman font for all plots
plt.rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 20

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

# Load the models and scalers
ys_model = joblib.load("YS_model(n).joblib")
ys_scaler = joblib.load("YS_scaler(n).joblib")
el_model = joblib.load("El_model(n).joblib")
el_scaler = joblib.load("El_scaler(n).joblib")

# Load the data and filter for Process = 100000
data = pd.read_csv('Data_set (reshuffled).csv')
filtered_data = data[data['Process'] == 100000].copy()

# Prepare the feature sets for each model
def prepare_features(df, selected_features):
    X = df.copy()
    for col in selected_features:
        if col not in X.columns:
            X[col] = 0
    return X[selected_features]

X_ys = prepare_features(filtered_data, selected_for_YS)
X_el = prepare_features(filtered_data, selected_for_El)

# Scale the features
X_ys_scaled = ys_scaler.transform(X_ys)
X_el_scaled = el_scaler.transform(X_el)

# Initialize SHAP explainers
explainer_ys = shap.Explainer(ys_model, X_ys_scaled, feature_names=selected_for_YS)
explainer_el = shap.Explainer(el_model, X_el_scaled, feature_names=selected_for_El)

# Calculate SHAP values
shap_values_ys = explainer_ys(X_ys_scaled)
shap_values_el = explainer_el(X_el_scaled)

# Create and save Swarm Plots
print("Creating high-quality SHAP Swarm Plots for Process = 100000...")

plt.figure(figsize=(12, 8), dpi=300)
shap.plots.beeswarm(shap_values_ys, max_display=15, show=False)
plt.tight_layout()
plt.savefig("YS_Swarm_Process_new.png", bbox_inches='tight', dpi=600)
plt.close()

plt.figure(figsize=(12, 8), dpi=300)
shap.plots.beeswarm(shap_values_el, max_display=15, show=False)
plt.tight_layout()
plt.savefig("El_Swarm_Process_new.png", bbox_inches='tight', dpi=600)
plt.close()

print("Successfully created high-quality swarm plots:")
print("- YS_Swarm_Process_new.png")
print("- El_Swarm_Process_new.png")
'''

with open("shap_main.py", "w") as f:
    f.write(code)

print("✅ SHAP script saved as 'shap_main.py'")
