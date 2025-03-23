# src/data_prep.py

import pandas as pd
import numpy as np
import os

def load_data(input_path):
    return pd.read_csv(input_path)

def clean_data(df):
    df = df.copy()

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df[df['TotalCharges'].notnull()]

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop irrelevant features
    columns_to_drop = ['customerID', 'gender', 'PhoneService']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df

def encode_features(df):
    # One-hot encode and convert bools to integers
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.astype(int)

    # Move 'Churn' column to the end
    if 'Churn' in df_encoded.columns:
        churn_col = df_encoded.pop('Churn')
        df_encoded['Churn'] = churn_col

    return df_encoded

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")

def main():
    input_path = "data/raw/telco_churn_raw.csv"
    output_path = "data/processed/telco_churn_processed.csv"

    print("📥 Loading raw data...")
    df = load_data(input_path)

    print("🧹 Cleaning data...")
    df_cleaned = clean_data(df)

    print("🔢 Encoding features...")
    df_encoded = encode_features(df_cleaned)

    print("💾 Saving processed data...")
    save_data(df_encoded, output_path)

if __name__ == "__main__":
    main()
