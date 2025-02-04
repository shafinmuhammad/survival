
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

filename = 'model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('survived Prediction App')
st.subheader('Please enter your data:')

df = pd.read_csv('passenger_survival_dataset.csv')
columns_list = df.columns.to_list()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    object_columns = df.select_dtypes(include=['object']).columns
    
    le = LabelEncoder()
    df['Gender'] = le.transform(df['Gender'])
    
    df['Class'] = df['Class'].replace({'First': 1, 'Business': 2, 'Economy': 3})
    df['Seat_Type'] = df['Seat_Type'].replace({'Window': 1, 'Middle': 2, 'Aisle': 3})


    prediction = loaded_model.predict(df)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.write(prediction_text)
