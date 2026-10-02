import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import shapely

from joblib import load

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL


@st.cache_data
def load_clean_data():
    return pd.read_parquet(DADOS_LIMPOS)


@st.cache_data
def load_geo_data():
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons into individual polygons
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Function to check and fix invalid geometries
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Fix invalid geometry
        # Orient the polygon to be counter-clockwise if it's a Polygon or MultiPolygon
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Apply the fix and orientation function to geometries
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Extract polygon coordinates
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Apply the coordinate conversion and store in a new column
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo


@st.cache_resource
def load_model():
    return load(MODELO_FINAL)


df = load_clean_data()
gdf_geo = load_geo_data()
model = load_model()


st.title("California Housing Price Prediction")

counties = list(gdf_geo["name"].unique())
counties.sort()

column1, column2 = st.columns(2)

with column1:

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

    st.write(df_input_model[['longitude', 'latitude', 'ocean_proximity']])

    button_price_prediction = st.button("Predict Price")

    if button_price_prediction:
        predicted_price = model.predict(df_input_model)

        st.write(
            f"Predicted Price: US${predicted_price[0][0]:,.2f}"
        )

with column2:
    view_state = pdk.ViewState(
        longitude=float(longitude),
        latitude=float(latitude),
        zoom=7,
        min_zoom=5,
        max_zoom=15,
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        gdf_geo[['geometry', 'name']],
        get_polygon="geometry",
        get_fill_color=[0, 0, 255, 100],
        get_line_color=[255, 255, 255],
        get_line_width=200,
    )

    selected_county_highlight_layer = gdf_geo.query('name == @selected_county')

    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=selected_county_highlight_layer[['geometry', 'name']],
        get_polygon="geometry",
        get_fill_color=[255, 0, 0, 150],
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        'html': '<b>County:</b> {name}',
        'style': {
            'backgroundColor': 'steelblue',
            'color': 'white',
            'fontsize': '10px'
        }
    }

    map = pdk.Deck(
        initial_view_state=view_state,
        map_style='light',
        layers=[polygon_layer, highlight_layer],
        tooltip=tooltip,
    )

    st.pydeck_chart(map)
