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
gdf_geo = load_geo_data()
model = load_model()


st.title("California Housing Price Prediction")

counties = list(gdf_geo["name"].sort_values())
selected_county = st.selectbox("County", counties)

# Seleciona somente uma linha do county escolhido
county_data = gdf_geo[gdf_geo["name"] == selected_county].iloc[0]

longitude = county_data["longitude"]
latitude = county_data["latitude"]

housing_median_age = st.number_input(
    "Housing Median Age",
    value=10,
    min_value=1,
    max_value=50
)

total_rooms = county_data["total_rooms"]
total_bedrooms = county_data["total_bedrooms"]
population = county_data["population"]
households = county_data["households"]

median_income = st.slider(
    "Median Income",
    min_value=5.0,
    max_value=100.0,
    value=45.0,
    step=5.0
)

ocean_proximity = county_data["ocean_proximity"]

bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
median_income_cat = np.digitize(
    median_income / 10,
    bins=bins_income
)

rooms_per_household = county_data["rooms_per_household"]
bedrooms_per_room = county_data["bedrooms_per_room"]
population_per_household = county_data["population_per_household"]


input_model = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income / 10,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household
}

df_input_model = pd.DataFrame([input_model])

st.write(df_input_model)

button_price_prediction = st.button("Predict Price")

if button_price_prediction:
    predicted_price = model.predict(df_input_model)

    st.write(
        f"Predicted Price: US${predicted_price[0][0]:,.2f}"
    )