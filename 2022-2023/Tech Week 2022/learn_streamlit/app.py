import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

#Config
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Food Demand Forecasting")

# For smooth and fast functioning
@st.cache
def load_data(nrows):
    data = pd.read_csv('train.csv', nrows=nrows)
    return data


@st.cache
def load_center_data(nrows):
    data = pd.read_csv('fulfilment_center_info.csv', nrows=nrows)
    return data


@st.cache
def load_meal_data(nrows):
    data = pd.read_csv('meal_info.csv', nrows=nrows)
    return data


# Printing in Streamlit to load data
data_load_state = st.text('Loading data...')
# Loading Weekly Data
weekly_data = load_data(1000)
# Loading Centre Info Data
center_info_data = load_center_data(1000)
# Loading Meal Data
meal_data = load_meal_data(1000)

# Bar Chart
st.subheader("Weekly Demand Data")
st.write(weekly_data)
st.bar_chart(weekly_data["num_orders"])

# histogram
df = pd.DataFrame(weekly_data[:200], columns=["num_orders", "checkout_price", "base_price"])
df.hist()
plt.show()
st.pyplot()

# Line Chart
st.line_chart(df)

# Area Chart
chart_data = pd.DataFrame(weekly_data[:40], columns=["num_orders", "base_price"])
st.area_chart(chart_data)

# Displaying Raw Data from the csv files
st.subheader("Fulfillment Center Information")
if st.checkbox("Show Center Information data"):
    st.subheader("Center Information data")
    st.write(center_info_data)

# Using Streamlit plotly to see the distribution region code and center id
hist_data = [center_info_data["center_id"],center_info_data["region_code"]]
group_labels = ["Center Id", "Region Code"]
fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 25])
st.plotly_chart(fig, use_container_width=True)

# Meal Information
st.subheader('Meal Information')
st.write(meal_data)
st.bar_chart(meal_data["cuisine"])

#Bonus
agree = st.button("Click to see Categories of Meal")
if agree:
 st.bar_chart(meal_data["category"])

# Command to run
# streamlit run app.py