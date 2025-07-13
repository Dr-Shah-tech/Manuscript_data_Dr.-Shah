# Manuscript_data_Dr.-Shah
This repository contains data and related code related to manuscript "Data-driven design of Al–Fe alloys for laser powder bed fusion to enhance recycled aluminum utilization"

Folder Structure

📁 data/
Contains datasets used in the study in CSV format.

Data_set (previous).csv
Dataset used in the previous study.

Data_set (updated).csv
Updated dataset including Al–Fe and Al–Mg₂Si alloys.

Al_Fe_alloys_literature-data.csv
Experimentally studied Al–Fe alloys from the literature.

Elemental_information.csv
Elemental properties of Al and alloying elements.



📁 scripts/
Python scripts for training models, predicting mechanical properties, generating new alloy compositions, and performing feature importance analysis:

Training_and_saving_GBR.py
Trains and saves Gradient Boosting Regression (GBR) models for yield strength (YS) and elongation (El). Feature selection is based on the previous study.

Predict_Al-Fe_reported_alloys.py
Uses trained GBR models to predict YS and El of Al–Fe alloys from Al_Fe_alloys_literature-data.csv.

generate_new_Al-Fe_based_alloys.py
Generates ~5000 Al–Fe based alloy compositions with Fe content between 1–5% using Random Forest Regression (RFR).

Compute_weighted_elemental_properties_of_new_Al-Fe_alloys.py
Calculates weighted elemental properties for the newly generated alloy compositions.

Predict_tensile_properties_of_new_Al-Fe_based_alloys.py
Predicts YS and El for the generated 5000 compositions using the pretrained GBR models.

SHAP_analysis.py
Performs SHAP (SHapley Additive exPlanations) analysis to evaluate feature importance in YS and El predictions.

bubble_plot_normalized_element_impact.py
Generates bubble plots to visualize the overall impact of each element on YS and El.
