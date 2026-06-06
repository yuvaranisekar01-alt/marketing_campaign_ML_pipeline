import joblib
import numpy as np
import pandas as pd

# sample data
sample_data = pd.DataFrame({
    'Campaign_Type': ['Email'],
    'Brand': ['Tira'],
    'Channel_Used': ['email, facebook'],

    'Cost_per_click': [4.5],
    'Acquisition_Cost': [50000],
    'Conversion_rate': [0.12] })

# load artifacts
reg_model = joblib.load("models/best_model_reg.pkl")
cls_model = joblib.load("models/best_model_clf.pkl")

scaler = joblib.load("models/scaler_reg.pkl")
ohe = joblib.load("models/ohe_clf.pkl")
mlb = joblib.load("models/mlb_clf.pkl")
le = joblib.load("models/label_encoder.pkl")

# encoding sample data
sample_ohe = pd.DataFrame(
    ohe.transform(sample_data[['Campaign_Type', 'Brand']]),
    columns= ohe.get_feature_names_out())

sample_mlb = pd.DataFrame(
    mlb.transform(sample_data['Channel_Used'].str.split(', ')),
    columns= mlb.classes_ )

# scaling sample data
num_cols= ['Cost_per_click', 'Acquisition_Cost', 'Conversion_rate']
sample_scaler = pd.DataFrame(
    scaler.transform(sample_data[num_cols]),
    columns = num_cols, index= sample_data.index
)

# sample final data
sample_final = pd.concat([sample_ohe, sample_mlb, sample_scaler], axis=1)

expected_cols = reg_model.get_booster().feature_names
sample_final = sample_final[expected_cols]


# regression- predict revenue
revenue_pred = reg_model.predict(sample_final)

acquisition_cost = sample_data['Acquisition_Cost'].iloc[0]
predicted_roi = ((revenue_pred[0]- acquisition_cost)/ acquisition_cost)
predicted_outcome = ("Profit" if predicted_roi > 0 else "Loss")

print(f"Predicted final revenue:, ₹{revenue_pred[0]:,.2f}")
print(f"Predicted ROI     : {predicted_roi[0]:,.2%}")
print(f"Predicted Outcome : {predicted_outcome}")

# Classification - predict Profit/Loss
cls_pred = cls_model.predict(sample_final)
profit_label = le.inverse_transform(cls_pred)
print(f"Predicted Outcome: {profit_label[0]}")


cls_pred = cls_model.predict(sample_final)
profit_label = le.inverse_transform(cls_pred)

cls_prob = cls_model.predict_proba(sample_final)

print(f"Classifier Prediction : {profit_label[0]}")
print(f"Classifier Probabilities :")

for label, prob in zip(le.classes_, cls_prob[0]):
    print(f"{label}: {prob:.2%}")

    
