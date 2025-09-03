# Demand Forecasting - Rossmann Store Sales

End-to-End Demand Forecasting Project on Rossmann Store Sales Dataset with Streamlit Deployment

Problem Statement : Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied. Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. By helping Rossmann create a robust prediction model, you will help store managers stay focused on what’s most important to them: their customers and their teams! 

Dataset Description : You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.

Link to Dataset : https://www.kaggle.com/competitions/rossmann-store-sales/overview

Files are described as below :
train.csv - historical data including Sales
test.csv - historical data excluding Sales
store.csv - supplemental information about the stores

Data fields : Most of the fields are self-explanatory. The following are descriptions for those that aren't.

Id - an Id that represents a (Store, Date) duple within the test set
Store - a unique Id for each store
Sales - the turnover for any given day (this is what you are predicting)
Customers - the number of customers on a given day
Open - an indicator for whether the store was open: 0 = closed, 1 = open
StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
StoreType - differentiates between 4 different store models: a, b, c, d
Assortment - describes an assortment level: a = basic, b = extra, c = extended
CompetitionDistance - distance in meters to the nearest competitor store
CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
Promo - indicates whether a store is running a promo on that day
Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

Project Overview : This project is an end-to-end machine learning pipeline for predicting daily sales of Rossmann stores. The goal is to forecast store sales for the next 6 weeks based on historical data, promotions, holidays, and store-specific information. The project includes:

Exploratory Data Analysis (EDA): Understand data distribution, identify trends and anomalies.
Feature Engineering: Generate meaningful features like day, month, week, promotions, store type, and one-hot encode categorical variables.
Model Training: Train an XGBoost regression model on historical sales data and save the model for later forecasting.
Forecasting: Generate daily sales predictions for future periods using a trained model.
Streamlit App: A frontend interface (planned) to allow users to upload minimal input (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday) and get forecasts.

The project structure follows best practices for reproducibility and modularity.

Environment Setup and Running the Code
1. Clone the Repository
git clone https://github.com/nikunjkichambare/demand_forecasting_rossmann.git
cd demand_forecasting_rossmann

2. Set Up Python Environment
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

Ensure requirements.txt includes packages like pandas, numpy, joblib, xgboost, streamlit.

3. Prepare Data

Place raw input CSVs in data/raw/ (example: test.csv).
Processed files will be saved in data/processed/.

4. Run Scripts
4.1) Feature Engineering & Model Training - python src/train_model.py

Generates features from raw data.
Trains XGBoost regression model.
Saves trained model to models/xgb_random_search_model.pkl.
Saves feature list to models/feature_list.txt.

4.2) Forecasting - python src/forecast.py

Reads data/raw/test.csv (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday).
Generates daily & weekly  forecasts for the next 6 weeks.
Dates are formatted as DD-MM-YYYY.
Negative predictions are capped at 0.
Saves output CSV to models/test_predictions.csv.

4.3) Streamlit App - streamlit run app/app.py

Allows users to upload CSV with minimal input (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday).
Generates forecasts using forecast.py.

Scripts Description

src/train_model.py : Trains the XGBoost model on processed data	Outputs Model (models/xgb_random_search_model.pkl) & Train Data Feature List (models/feature_list.txt)
src/forecast.py : Generates 6-week daily sales forecasts	Reads raw CSV (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday), formats date as DD-MM-YYYY, caps negative predictions
app/app.py : Streamlit frontend Provides user interface for uploading CSVs and viewing forecasts
notebooks/ : EDA and experimentation, Visualizations, missing value handling, feature exploration

Project Pipeline Diagram (src/run_pipeline.py) : Execute from root via "python -m src.run_pipeline"

Raw Data (data/raw/)
      │
      ▼
Exploratory Data Analysis (notebooks/01_eda.ipynb)
      │
      ▼
Feature Engineering (notebooks/02_feature_engineering.ipynb)
      │
      ▼
Data Ingestion (src/data_ingestion.py)
      │
      ▼
Data Validation (src/data_validation.py)
      │
      ▼
XGBoost Model Training (src/train_model.py) → Save Model (models/xgb_random_search_model.pkl) & Train Data Feature List (models/feature_list.txt)
      │
      ▼
Forecasting (src/model_predictions.py) → Save Predictions on Test Data (models/test_predictions.csv) 
      │
      ▼
Model Performance Evaluations (src/evaluate_model.py) → Save Performance Metrics (models/model_performance.csv)
      │
      ▼
Streamlit App (app/app.py)
  └─ User uploads CSV file with Minimal Data : test.csv (Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday)
  └─ Generates and displays forecasts

Explanation

Raw Data: Historical store sales and store information.
EDA: Identify trends, missing values, and insights.
Feature Engineering: Transform raw data into model-ready features.
Model Training: Train XGBoost regression model using processed features.
Forecasting: Predict future sales for each store.
Streamlit App: Frontend interface for users to generate forecasts easily.