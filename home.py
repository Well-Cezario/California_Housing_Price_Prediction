import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

from joblib import load

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL


@st.cache_data
def load_clean_data():
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data
def load_geo_data():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)


@st.cache_resource
def load_model():
    return load(MODELO_FINAL)


df = load_clean_data()
gdf = load_geo_data()
model = load_model()


st.title('California Housing Price Prediction')

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)

housing_median_age = st.number_input("Housing Median Age", value=10)

total_rooms = st.number_input("Total Rooms", value=800)
total_bedrooms = st.number_input("Total Bedrooms", value=100)
population = st.number_input("Population", value=300)
households = st.number_input("Households", value=100)

median_income = st.slider("Median Income", min_value=0.5,
                          max_value=15.0, value=4.0, step=0.5)

ocean_proximity = st.selectbox(
    "Ocean Proximity", df['ocean_proximity'].unique())

median_income_cat = st.selectbox(
    "Median Income Category", options=[1, 2, 3, 4, 5, 6])

rooms_per_household = st.number_input("Rooms per Household", value=7)
bedrooms_per_room = st.number_input("Bedrooms per Room", value=0.2)
population_per_household = st.number_input("Population per Household", value=2)

input_model = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household
}

df_input_model = pd.DataFrame([input_model])

button_price_prediction = st.button("Predict Price")

if button_price_prediction:
    predicted_price = model.predict(df_input_model)
    st.write(f"Predicted Price: US${predicted_price[0][0]:,.2f}")
