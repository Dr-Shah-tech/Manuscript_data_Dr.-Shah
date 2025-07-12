# Compute_weighted_elemental_properties_of_new Al-Fe alloys.py

import pandas as pd
import numpy as np

# Load the alloy compositions (each row is an alloy, columns are elements with fractional values)
compositions_df = pd.read_csv('5000_new Al-Fe based alloys.csv')

# Load elemental properties data (row: property, column: element)
elemental_info_df = pd.read_csv('Elemental_information.csv')

# Set 'Property' as the index and ensure all values are numeric (NaN for non-convertibles)
elemental_info_df.set_index('Property', inplace=True)
elemental_info_df = elemental_info_df.apply(pd.to_numeric, errors='coerce')

# Extract elements from composition DataFrame
elements = compositions_df.columns.tolist()

# Prepare a DataFrame to store composition-weighted properties
weighted_properties_df = pd.DataFrame()

# Compute weighted average for each property
for property_name in elemental_info_df.index:
    # Get property values for relevant elements
    property_values = elemental_info_df.loc[property_name]

    # Align to elements present in compositions_df
    property_values = property_values.reindex(elements).fillna(0)

    # Calculate weighted average: sum(C_i * X_i)
    weighted_average = compositions_df.mul(property_values, axis=1).sum(axis=1)

    # Store results
    weighted_properties_df[property_name] = weighted_average

# Combine original compositions with computed weighted properties
result_df = pd.concat([compositions_df, weighted_properties_df], axis=1)

# Save results
result_df.to_csv('alloy_with_weighted_elemental_properties.csv', index=False)

# Preview
print(result_df.head())
