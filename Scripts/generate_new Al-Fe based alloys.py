# Generate_Al-Fe based new alloys.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import time

# Define composition bounds (must sum to 1)
bounds = {
    'Si': (0.0, 0.2),
    'Fe': (0.01, 0.05),
    'Cu': (0.0, 0.1),
    'Mg': (0.0, 0.15),
    'Zn': (0.0, 0.1),
    'Ti': (0.0, 0.03),
    'Zr': (0.0, 0.015),
    'Sc': (0.0, 0.015),
    'Mn': (0.0, 0.05),
    'Al': (0.85, 1.0)  # Adjusted to allow sum=1
}

# Load and filter data
data = pd.read_csv('data_for_BO.csv')
data = data[(data['Process'] == 100000) & 
           (data['SHT_Temperature'] == 0) &
           (data['SHT_Time'] == 0) &
           (data['Aging_temperature'] == 0) &
           (data['Aging_Time'] == 0)]

# Prepare features and targets
features = list(bounds.keys())
X = data[features].values
y_ys = data['YS'].values
y_el = data['El'].values

# Train models
print("Training models...")
start_time = time.time()
model_ys = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_el = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_ys.fit(X, y_ys)
model_el.fit(X, y_el)
print(f"Models trained in {time.time()-start_time:.1f} seconds")

def generate_normalized_composition():
    """Generate composition that sums exactly to 1"""
    while True:
        comp = np.array([np.random.uniform(low, high) for (low, high) in bounds.values()])
        comp /= comp.sum()  # Normalize to sum=1
        
        # Verify all elements still within bounds after normalization
        if all(bounds[feat][0] <= comp[i] <= bounds[feat][1] 
               for i, feat in enumerate(features)):
            return comp

# Generate 5000 valid alloys
n_alloys = 5000
valid_alloys = []
print(f"\\nGenerating {n_alloys} valid compositions (sum=1)...")

for _ in tqdm(range(n_alloys)):
    # Keep generating until we get a valid alloy
    while True:
        comp = generate_normalized_composition()
        ys_pred = model_ys.predict([comp])[0]
        el_pred = model_el.predict([comp])[0]
        
        if (400 <= ys_pred <= 600) and (5 <= el_pred <= 15):
            valid_alloys.append({
                **dict(zip(features, comp)),
                'Process': 100000,
                'SHT_Temperature': 0,
                'SHT_Time': 0,
                'Aging_temperature': 0,
                'Aging_Time': 0,
                'YS_pred': ys_pred,
                'El_pred': el_pred,
                'Sum': comp.sum()  # Verification (should be 1.0)
            })
            break

# Create DataFrame and save
df = pd.DataFrame(valid_alloys)
df.to_csv('5000_new Al-Fe based alloys.csv', index=False)

# Verification
print("\\nVerification:")
print(f"Generated {len(df)} alloys")
print("Composition sums:")
print(f"Min: {df[features].sum(axis=1).min():.6f}")
print(f"Max: {df[features].sum(axis=1).max():.6f}")
print("\\nProperty ranges:")
print(f"YS: {df['YS_pred'].min():.1f}-{df['YS_pred'].max():.1f} MPa")
print(f"El: {df['El_pred'].min():.1f}-{df['El_pred'].max():.1f} %")
print(f"\\nTotal time: {time.time()-start_time:.1f} seconds")
'''

# Save it to a Python file
with open("generate_alloys.py", "w") as f:
    f.write(code)

print("✅ Script saved as 'generate_alloys.py'")
