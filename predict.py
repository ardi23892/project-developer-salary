import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def model_load():
    with open("SavedModel.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = model_load()

regressor = data["model"]
CountryEncoder = data["Country_LE"]
EdLevelEncoder = data["EdLevel_LE"]

def predict_page():
    st.title('Predict Your Salary')
    st.write('Predict your salary as a developer based on your Country, Education Level, and Numbers of Years in Professional Coding.')

    countries = (
        'Sweden',
        'Spain',
        'Germany',
        'Canada',
        'France',
       'United Kingdom of Great Britain and Northern Ireland',
       'Russian Federation',
       'United States of America',
       'Italy',
       'Netherlands',
       'Poland',
       'Australia',
       'India',
       'Brazil'
    )

    educations = (
        "Master's degree",
        "Bachelor's degree",
       'Professional or Doctoral Degree',
       'Less than a Bachelor'
    )

    country = st.selectbox("Country", countries)

    education = st.selectbox("Education Level", educations)

    yearspro = st.slider("Numbers of Years in Professional Coding", 0,25,1)

    submit = st.button("Predict Your Salary")
    if submit:
        x = np.array([[country,education, yearspro]])
        x[:,0] = CountryEncoder.transform(x[:,0])
        x[:,1] = EdLevelEncoder.transform(x[:,1])
        x = x.astype(float)

        predicted = regressor.predict(x)
        st.subheader(f"The estimated salary is ${predicted[0]:.2f}")
    
    df = data_load()

    st.write("Mean Salary Based On Country")

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("Mean Salary Based On Years of Professional Coding")

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

def CutCountry(categories, minimum):
    NewCountry = {}
    for i in range(len(categories)):
        if categories.values[i] >= minimum:
            NewCountry[categories.index[i]] = categories.index[i]
        else:
            NewCountry[categories.index[i]] = 'Other'
    return NewCountry

def StF(x):
    if x == "More than 50 years":
        return 51
    if x == "Less than 1 year":
        return 0.5
    return float(x)

def CombineEdLevel(x):
    if "Bachelor" in x:
        return "Bachelor's degree"
    if "Master" in x:
        return "Master's degree"
    if "Professional" in x or "Other doctoral" in x:
        return "Professional or Doctoral Degree"
    return "Less than a Bachelor"

@st.cache
def data_load():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country","EdLevel","YearsCodePro","Employment","ConvertedCompYearly"]]
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)
    CuttedCountry = CutCountry(df.Country.value_counts(),500)
    df['Country'] = df['Country'].map(CuttedCountry)
    df = df[df["Country"] != "Other"]
    df = df[df["Salary"] <= 250000]
    df = df[df["Salary"] >= 10000]

    df["YearsCodePro"] = df["YearsCodePro"].apply(StF)
    df["EdLevel"] = df["EdLevel"].apply(CombineEdLevel)
    return df