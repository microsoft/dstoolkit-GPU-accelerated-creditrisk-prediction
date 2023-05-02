import streamlit as st
from streamlit_shap import st_shap
import pandas as pd
import joblib
import shap
# Load data
data = pd.read_csv("X_test_df.csv")
data = data.iloc[:2000]
#load model
import pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
#load shap values
shap_values = joblib.load("shap_values.joblib")
shap.initjs()
shap_values = shap_values[0:2000]
data  = data.iloc[:, : -14]
# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(data)
# Create app
def app():
    # Add a title
    st.title("My Streamlit App")
    
    # Add some text
    st.write("This is a simple example of a Streamlit app.")
    
    # Show data
    st.write("Here's some data:")
    st.write(data.iloc[:5])
    
    # Explain prediction
    shap_values = explainer(data)
    st.write("Here's an explanation of the first prediction:")
    #st_shap(shap.plots.waterfall(shap_values), height=300)
    #summary_plot_bar
    #summary_plot_bar
#shap.summary_plot(shap_values, X, plot_type="bar")
    #st_shap(shap.summary_plot(shap_values, data, plot_type="bar"), height=500)
    st_shap(shap.force_plot(explainer.expected_value, shap_values.values[0,:], data.iloc[0,:],show=False,matplotlib=True), height=200, width=1000)
    #st.pyplot()
    
if __name__ == '__main__':
    app()