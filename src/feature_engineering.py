import pandas as pd
import numpy as np
from utils import numerical_cols, categorical_cols, target_reg, target_cls
import sys
import os

def load_data():
    #loading cleaned dataset
    df = pd.read_csv(r"data/processed/clean_df.csv")

    #converting data types
    df['ROI'] = pd.to_numeric(df['ROI'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

def create_features(df) ->pd.DataFrame:

    #Extracting month and year from date column
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter

    #creating new features/target column based on ROI_manual
    df['Profit_loss'] = df['ROI'].apply(lambda x: 'Profit' if x > 0 else 'Loss')

    #ratio features
    df['CTR'] = df['Clicks']/df['Impressions'].replace(0, np.nan)    #Click-Through Rate
    df['Conversion_rate'] = df['Conversions']/ df['Clicks'].replace(0, np.nan)    #how many clicks became conversions
    df['Lead_Rate'] = df['Leads']/ df['Clicks'].replace(0, np.nan)    #how many click became leads
    df['Revenue_per_lead'] = df['Revenue']/ df['Leads'].replace(0, np.nan)    #revenue generated per lead
    df['Cost_per_click'] = df['Acquisition_Cost']/ df['Clicks'].replace(0, np.nan)   #cost incurred per click

    #filling NaN values 
    ratio_cols = ['CTR', 'Conversion_rate', 'Lead_Rate', 'Revenue_per_lead', 'Cost_per_click']
    df[ratio_cols] = df[ratio_cols].fillna(0)

    #drop date column as we have extracted month, year and quarter from it
    df = df.drop('Date', axis=1)

    #drop ROI column to prevent overfitting
    df = df.drop('ROI', axis=1)
    
    return df

def save_data(df):
     df.to_csv(r"data/processed/feature_engineered_df.csv", index=False )
     print("✅ Feature engineering completed and saved to data/processed/feature_engineered_df.csv")
     print("New features created: Month, Year, Quarter, Profit_loss, CTR, Conversion_rate, Lead_Rate, Revenue_per_lead, Cost_per_click")
     print(f"target distribution:\n{(df['Profit_loss'].value_counts(normalize=True)*100).round(2)}%")


if __name__ == "__main__":

    df = load_data()
    df = create_features(df)
    save_data(df)


