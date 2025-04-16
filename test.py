import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



url = "https://github.com/impolichetti/airbnb_clone/raw/refs/heads/main/airbnb_clone/airbnb/used_cars.csv"
cars_df = pd.read_csv(url)

st.title("Cars Data Visualization")
column = st.selectbox("Select a column to visualize", cars_df.columns)
if cars_df[column].dtype == 'object' or cars_df[column].nunique() < 50:
    st.subheader(f"Value counts of {column}")
    value_counts = cars_df[column].value_counts()
    st.bar_chart(value_counts)
else:
    st.write(f"The column '{column}' is numeric. Showing histogram.")
    st.hist(cars_df[column], bins=20)