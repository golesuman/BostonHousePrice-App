from numpy.core.numeric import indices
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor

st.write("""
# Simple House Price Predictor
## This app predicts the price of house in Boston.
""")
st.sidebar.header("User Input Parameter.")

def user_input():
    CRIM=st.sidebar.number_input("Crime rate per town")
    ZN=st.sidebar.number_input("Land Zoned")
    INDUS=st.sidebar.number_input("Non retail business per town.")
    NOX=st.sidebar.number_input("Nitric Oxides Concentration")
    RM=st.sidebar.number_input("Average number of rooms per dwelling")
    AG=st.sidebar.number_input(" proportion of owner-occupied units built prior to 1940")
    RAD=st.sidebar.number_input("Index of accessibility to radial Highways")
    TAX=st.sidebar.number_input("Tax rate per $10000")
    PTRATIO=st.sidebar.number_input("Teacher pupil per town")
    B=st.sidebar.number_input("Proportion of black by town ")
    LSTAT=st.sidebar.number_input("Lower status of population")
    data={
        'CRIM':CRIM,
        'ZN':ZN,
        'INDUS':INDUS,
        'NOX':NOX,
        'RM':RM,
        'AG':AG,
        'RAD':RAD,
        'TAX':TAX,
        'PTRATIO':PTRATIO,
        'B':B,
        'LSTAT':LSTAT
        }
    features=pd.DataFrame(data, index=[0])
    return features
df_features=user_input()
st.subheader("User input Parameters")
st.write(df_features)
dataset=load_boston()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['SellingPrice']=dataset.target
df.drop(['CHAS','DIS'],inplace=True,axis='columns')
x=df.drop('SellingPrice',axis=1)
y=df['SellingPrice']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3,test_size=0.2)
model=AdaBoostRegressor(n_estimators=60,learning_rate=0.4)
model.fit(x_train,y_train)
prediction=model.predict(df_features)
st.subheader("Price Prediction(per $1000)")
st.write(prediction)


