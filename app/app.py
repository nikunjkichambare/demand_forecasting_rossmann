import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import joblib

# Define paths dynamically relative to this app.py file location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'store_feature_engineered.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_random_search_model.pkl')
FEATURE_LIST_PATH = os.path.join(BASE_DIR, 'models', 'feature_list.txt')


@st.cache_resource
def load_store_data():
    try:
        return pd.read_csv(STORE_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Store data file not found at {STORE_DATA_PATH}. Please check the data folder.")
        raise


@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please check the models folder.")
        raise
    try:
        with open(FEATURE_LIST_PATH, 'r') as f:
            feature_list = [line.strip() for line in f]
    except FileNotFoundError:
        st.error(f"Feature list file not found at {FEATURE_LIST_PATH}. Please check the models folder.")
        raise
    return model, feature_list


def preprocess_input(df_raw, store_df, feature_list):
    df = df_raw.copy()

    df['Open'].fillna(1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

    cat_cols = ['DayOfWeek', 'Open', 'Promo', 'SchoolHoliday']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_ohe = ohe.fit_transform(df[cat_cols])
    cat_ohe_df = pd.DataFrame(cat_ohe, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), cat_ohe_df], axis=1)

    ohe_sh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    stateholiday_ohe = ohe_sh.fit_transform(df[['StateHoliday']])
    stateholiday_df = pd.DataFrame(stateholiday_ohe, columns=ohe_sh.get_feature_names_out(['StateHoliday']), index=df.index)
    df = pd.concat([df.drop(columns=['StateHoliday']), stateholiday_df], axis=1)

    df.drop(columns=['Date'], inplace=True)

    df = df.merge(store_df, how='left', on='Store')

    df.fillna(0, inplace=True)

    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_list]

    return df


def generate_future_dates(store_df, weeks=6, start_date=None):
    if start_date is None:
        start_date = pd.Timestamp.today().normalize()

    future_dates = [start_date + pd.Timedelta(days=i) for i in range(weeks * 7)]
    stores = store_df['Store'].unique()

    future_list = []
    for store in stores:
        for dt in future_dates:
            future_list.append({'Store': store, 'Date': dt})

    future_df = pd.DataFrame(future_list)

    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Day'] = future_df['Date'].dt.day
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek + 1
    future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week.astype(int)
    future_df['Quarter'] = future_df['Date'].dt.quarter
    future_df['IsWeekend'] = future_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

    future_df['Open'] = 1
    future_df['Promo'] = 0
    future_df['SchoolHoliday'] = 0
    future_df['StateHoliday'] = '0'

    return future_df


def main():
    st.title("Sales Prediction App with 6-Week Future Forecast")

    forecast_start_date = st.date_input(
        "Select forecast start date (default today)",
        value=pd.Timestamp.today().date()
    )
    st.write("Selected date (DD-MM-YYYY):", forecast_start_date.strftime("%d-%m-%Y"))

    store_df = load_store_data()
    model, feature_list = load_model_and_features()

    uploaded_file = st.file_uploader(
        "Upload CSV with raw features (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday)", 
        type=['csv']
    )
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Raw Input Preview:", raw_df.head())

        try:
            X = preprocess_input(raw_df, store_df, feature_list)
            preds_log = model.predict(X)
            preds = np.expm1(preds_log)

            result_df = pd.DataFrame({
                'Id': raw_df.get('Id', pd.Series(range(len(raw_df)))),
                'Sales_Prediction': preds
            })

            st.subheader("Predictions on Uploaded Data")
            st.write(result_df)

            future_df = generate_future_dates(store_df, weeks=6, start_date=pd.Timestamp(forecast_start_date))
            st.write(f"Generating future features for {len(future_df)} rows")

            future_X = preprocess_input(future_df, store_df, feature_list)

            future_preds_log = model.predict(future_X)
            future_preds = np.expm1(future_preds_log)

            future_df['Sales_Prediction'] = future_preds
            future_df['Date'] = future_df['Date'].dt.strftime('%d-%m-%Y')

            st.subheader("6-Week Daily Sales Forecast")
            st.write(future_df[['Store', 'Date', 'Sales_Prediction']].head(20))

            weekly_forecast = future_df.groupby(['Store', 'WeekOfYear'])['Sales_Prediction'].mean().reset_index()
            weekly_forecast['Forecast_Week_Start'] = weekly_forecast['WeekOfYear'].apply(
                lambda w: pd.Timestamp.fromisocalendar(forecast_start_date.year, w, 1)
            )
            weekly_forecast['Forecast_Week_Start'] = weekly_forecast['Forecast_Week_Start'].dt.strftime('%d-%m-%Y')

            st.subheader("6-Week Average Weekly Sales Forecast per Store")
            st.write(weekly_forecast.head(20))

            csv_future = future_df.to_csv(index=False).encode()
            st.download_button("Download 6-Week Daily Forecast CSV", csv_future, "6_week_daily_forecast.csv", "text/csv")

            csv_weekly = weekly_forecast.to_csv(index=False).encode()
            st.download_button("Download 6-Week Weekly Average Forecast CSV", csv_weekly, "6_week_weekly_avg_forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during preprocessing or prediction: {e}")


if __name__ == "__main__":
    main()
