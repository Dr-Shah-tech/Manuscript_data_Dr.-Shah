# bubble_plot_normalized_element_impact.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Set fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Load data
data = pd.read_csv('predicted_YS_El_of_new Al-Fe alloys.csv')
X = data.drop(['Predicted_YS', 'Predicted_El'], axis=1)
y_ys = data['Predicted_YS']
y_el = data['Predicted_El']

# Calculate correlations
corr_ys = X.corrwith(y_ys)
corr_el = X.corrwith(y_el)

# Normalize to [0,1] for comparable scales
scaler = MinMaxScaler()
impact_ys = scaler.fit_transform(corr_ys.values.reshape(-1, 1)).flatten()
impact_el = scaler.fit_transform(corr_el.values.reshape(-1, 1)).flatten()

# Total importance (bubble size)
total_impact = (np.abs(corr_ys) + np.abs(corr_el)).values

# Create bubble plot
plt.figure(figsize=(10, 8), dpi=300)
scatter = plt.scatter(
    x=impact_ys, 
    y=impact_el, 
    s=total_impact*500,  # Scale bubble size
    c=total_impact, 
    cmap='seismic',
    alpha=0.7
)

# Annotate elements
for i, elem in enumerate(X.columns):
    plt.annotate(elem, (impact_ys[i]+0.02, impact_el[i]), ha='left')

# Reference lines
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

# Labels and title
plt.xlabel('Normalized Impact on Yield Strength (YS) →', fontweight='bold', fontsize=24)
plt.ylabel('Normalized Impact on Elongation (El) →', fontweight='bold', fontsize=24)
#plt.title('Combined Impact of Alloying Elements', pad=20, fontweight='bold')

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Total Impact (YS + El)', fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('combined_impact.png', bbox_inches='tight')
plt.show()
