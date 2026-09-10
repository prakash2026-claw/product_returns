import pandas as pd
from pycaret.classification import load_model, predict_model
import streamlit as st

# Set page configurations
st.set_page_config(page_title="Product Returns Predictor", layout="centered")

# 1. Load the trained PyCaret model (cached so it only loads once)
@st.cache_resource
def get_model():
    # Make sure model is in the same directory as this script
    return load_model("product_returns")

model = get_model()

st.title("🏥 Product Returns Prediction Dashboard")
st.write("Fill out the product,client and worker realted details below to check the Product Returns prediction.")

# 2. Build the Form Interface
with st.form("prediction_form"):
    st.subheader("Categorical Inputs")
    col1 = st.columns(1)
    
    with col1:
        day_of_week = st.number_input("day_of_week", min_value=0.0, value=1
        quantity = st.number_input("quantity", min_value=0.0, value=1222
        complexity = st.number_input("complexity", min_value=0.0, value=1
        is_rush = st.number_input("is_rush", min_value=0.0, value=0
        electricity_issue = st.number_input("electricity_issue", min_value=0.0, value=0
        is_weekend = st.number_input("is_weekend", min_value=0.0, value=0
        primary_worker_skill = st.number_input("primary_worker_skill", min_value=0.0, value=0.841
        num_workers = st.number_input("num_workers", min_value=0.0, value=1
        avg_worker_skill = st.number_input("avg_worker_skill", min_value=0.0, value=0.841
        min_worker_skill = st.number_input("min_worker_skill", min_value=0.0, value=0.841

    st.subheader("Numerical Inputs")
    col2 = st.columns(1)
    
    with col2:
        date = st.text_input("date",  value="2026-06-16"
        shift = st.text_input("shift",  value="morning"
        product_type = st.text_input("product_type",  value="plastic_bag"
        material_type = st.text_input("material_type",  value="standard_cardboard"
        client_id = st.text_input("client_id",  value="C014"
        primary_worker_id = st.text_input("primary_worker_id",  value="W007"
        primary_worker_profile = st.text_input("primary_worker_profile",  value="experienced"


    # Submit button for the form
    submit_button = st.form_submit_button("Predict Product Returns")

# 3. Handle Prediction Logic upon form submission
if submit_button:
    # Compile the form inputs into a dictionary matching your PyCaret model's features
    input_data = {'date': date,'day_of_week': day_of_week,'shift': shift,'product_type': product_type,'material_type': material_type,'quantity': quantity,'complexity': complexity,'is_rush': is_rush,'electricity_issue': electricity_issue,'is_weekend': is_weekend,'client_id': client_id,'primary_worker_id': primary_worker_id,'primary_worker_skill': primary_worker_skill,'primary_worker_profile': primary_worker_profile,'num_workers': num_workers,'avg_worker_skill': avg_worker_skill,'min_worker_skill': min_worker_skill}
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])
    
    with st.spinner("Calculating risk..."):
        # Make the prediction using PyCaret
        predictions = predict_model(model, data=df,round=2,raw_score=True)
        prediction_label = predictions["prediction_label"].iloc[0]
        prediction_score_1 = predictions["prediction_score_1"].iloc[0]
        prediction_score_0 = predictions["prediction_score_0"].iloc[0]
        
        # Display the result to the user
        st.success("### Prediction Complete!")
        st.metric(label="Risk Status Result", value=f"Class: {prediction_label}")
        st.metric(label="Prediction Confidence Scores", value=f"Class 0 Score:{prediction_score_0}, Class 1 Score:{prediction_score_1}")
