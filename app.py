import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV from GitHub
url = "https://github.com/impolichetti/airbnb_clone/raw/refs/heads/main/airbnb_clone/airbnb/used_cars.csv"
cars_df = pd.read_csv(url)

# Title
st.title("Cars Data Visualization")

# User selects a column
column = st.selectbox("Select a column to visualize", cars_df.columns)

# Categorical or few unique values
if cars_df[column].dtype == 'object' or cars_df[column].nunique() < 50:
    st.subheader(f"Value counts of {column}")
    value_counts = cars_df[column].value_counts()
    st.bar_chart(value_counts)

# Numeric columns — show a histogram
else:
    st.subheader(f"Histogram of {column}")
    fig, ax = plt.subplots()
    ax.hist(cars_df[column].dropna(), bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)



