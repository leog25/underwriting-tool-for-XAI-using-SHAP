import streamlit as st
import pandas as pd
import numpy as np
import shap
from openai import OpenAI
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from sklearn.ensemble import RandomForestClassifier
from utils import generate_explanation_letter, create_sample_data, preprocess_input

openai = OpenAI(
    api_key='YOUR-OPENAI-KEY'
)

st.set_page_config(
    page_title="Underwriting Explainability Tool",
    page_icon="📋",
    layout="wide"
)

st.title("Underwriting Explainable AI (XAI) Tool using SHAP values with Random Forest Classifier model and OpenAI")
st.markdown("""
This tool helps underwriters understand and explain policy decisions using AI and machine learning insights.
""")

with st.sidebar:
    st.header("Policy Details")
    
    policy_number = st.text_input("Policy Number", "POL-2024-001")
    applicant_name = st.text_input("Applicant Name", "John Doe")
    
    st.subheader("Risk Factors")
    credit_score = st.slider("Credit Score", 300, 850, 700)
    age = st.number_input("Age", 18, 100, 35)
    income = st.number_input("Annual Income ($)", 0, 1000000, 50000, step=1000)
    claims_history = st.number_input("Previous Claims", 0, 10, 0)
    coverage_amount = st.number_input("Requested Coverage ($)", 0, 1000000, 100000, step=10000)
    
    decision = st.selectbox("Policy Decision", ["Approved", "Rejected"])
    
    analyze_button = st.button("Analyze and Generate Letter")


if analyze_button:

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Decision Analysis")
        

        X_train, y_train = create_sample_data()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        current_policy = preprocess_input({
            'credit_score': credit_score,
            'age': age,
            'income': income,
            'claims_history': claims_history,
            'coverage_amount': coverage_amount
        })
        

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(current_policy)
        

        shap_for_approval = shap_values[0][0]       
        shap_for_approval_inverted = -shap_for_approval 
        

        st.subheader("Feature Importance Analysis")
        fig = go.Figure()
        
        features = ['Credit Score', 'Age', 'Income', 'Claims History', 'Coverage Amount']
        values = shap_for_approval_inverted
        
        fig.add_trace(go.Bar(
            y=features,
            x=values,
            orientation='h',
 
            marker_color=['#1f77b4' if val > 0 else '#d62728' for val in values]
        ))
        
        fig.update_layout(
            title="Impact on Approval (Positive = Increases Approval Probability)",
            xaxis_title="SHAP Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.header("Explanation Letter")
        

        shap_values_for_letter = [arr.copy() for arr in shap_values]  
        shap_values_for_letter[0][0] = shap_for_approval_inverted    
        
        letter = generate_explanation_letter(
            applicant_name=applicant_name,
            policy_number=policy_number,
            decision=decision,
            shap_values=shap_values_for_letter,
            features=features,
            feature_values={
                'credit_score': credit_score,
                'age': age,
                'income': income,
                'claims_history': claims_history,
                'coverage_amount': coverage_amount
            }
        )
        

        st.markdown(letter)
        

        st.download_button(
            label="Download Letter (PDF)",
            data=letter,
            file_name=f"explanation_letter_{policy_number}.txt",
            mime="text/plain"
        )


st.markdown("---")
st.markdown("*Created by Leo G in hopes of aiding the future of AI poweredunderwriters in Insurance*")
