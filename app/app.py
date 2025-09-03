import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import joblib

@st.cache_resource
def load_store_data():
    # Load preprocessed store features once
    return pd.read_csv('C:\\Users\\nikun\\Documents\\Demand Forecasting\\data\\processed\\store_feature_engineered.csv')

@st.cache_resource
def load_model_and_features():
    model = joblib.load('C:\\Users\\nikun\\Documents\\Demand Forecasting\\models\\xgb_random_search_model.pkl')
    with open('C:\\Users\\nikun\\Documents\\Demand Forecasting\\models\\feature_list.txt', 'r') as f:
        feature_list = [line.strip() for line in f]
    return model, feature_list

def preprocess_input(df_raw, store_df, feature_list):
    df = df_raw.copy()

    # Handle missing Open if any
    df['Open'].fillna(1, inplace=True)
    
    # Parse Date and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # One-hot encode categorical columns
    cat_cols = ['DayOfWeek', 'Open', 'Promo', 'SchoolHoliday']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_ohe = ohe.fit_transform(df[cat_cols])
    cat_ohe_df = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), cat_ohe_df], axis=1)
    
    # One-hot encode StateHoliday separately
    ohe_sh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    stateholiday_ohe = ohe_sh.fit_transform(df[['StateHoliday']])
    stateholiday_df = pd.DataFrame(stateholiday_ohe, columns=ohe_sh.get_feature_names_out(['StateHoliday']), index=df.index)
    df = pd.concat([df.drop(columns=['StateHoliday']), stateholiday_df], axis=1)
    
    # Drop original Date column after extraction
    df.drop(columns=['Date'], inplace=True)

    # Merge with store features
    df = df.merge(store_df, how='left', on='Store')
    
    # Fill missing store data after merge
    df.fillna(0, inplace=True)

    # Ensure columns match model features exactly (fill missing with 0)
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_list]

    return df

def main():
    st.title("Sales Prediction App with Feature Engineering")

    uploaded_file = st.file_uploader("Upload CSV with raw features", type=['csv'])
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)

        store_df = load_store_data()
        model, feature_list = load_model_and_features()

        st.write("Raw Input Preview:", raw_df.head())

        try:
            X = preprocess_input(raw_df, store_df, feature_list)
            st.write("Processed Features for Model Input:", X.head())

            preds_log = model.predict(X)
            preds = np.expm1(preds_log)

            result_df = pd.DataFrame({
                'Id': raw_df.get('Id', pd.Series(range(len(raw_df)))),
                'Sales_Prediction': preds
            })

            st.write(result_df)

            # Provide option to download predictions
            csv = result_df.to_csv(index=False).encode()
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during preprocessing or prediction: {e}")

if __name__ == "__main__":
    main()
