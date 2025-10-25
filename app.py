import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🎯",
    layout="wide"
)

# Load the saved model and related files
@st.cache_resource
def load_model():
    pipeline = joblib.load('models/customer_segmentation_pipeline.joblib')
    cluster_labels = joblib.load('models/cluster_labels_mapping.joblib')
    features = joblib.load('models/feature_names.joblib')
    return pipeline, cluster_labels, features

# Main header
st.title("🎯 Customer Segmentation Analysis")
st.markdown("---")

try:
    pipeline, cluster_labels, features = load_model()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Single Customer Prediction")
        st.write("Enter customer details to predict their segment:")
        
        # Input fields
        gender = st.radio("Gender", options=["Female", "Male"])
        age = st.slider("Age", min_value=18, max_value=90, value=30)
        income = st.slider("Annual Income (k$)", min_value=0, max_value=200, value=50)
        spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
        
        if st.button("Predict Segment"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Annual Income (k$)': [income],
                'Spending Score (1-100)': [spending]
            })
            
            # Make prediction
            cluster = pipeline.predict(input_data)[0]
            segment = cluster_labels[cluster]
            
            # Show prediction with styled box
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #1f77b4; margin-bottom: 10px;'>Predicted Segment</h3>
                <h2 style='color: #2e7d32;'>{segment}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Batch Prediction")
        st.write("Upload a CSV file with multiple customers:")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Read and process the uploaded file
            df = pd.read_csv(uploaded_file)
            required_columns = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            
            if all(col in df.columns for col in required_columns):
                # Make predictions
                clusters = pipeline.predict(df)
                df['Segment'] = [cluster_labels[c] for c in clusters]
                
                # Show results
                st.write("Predictions:")
                st.dataframe(df)
                
                # Download button for results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="segmentation_results.csv",
                    mime="text/csv"
                )
                
                # Visualization
                st.subheader("Segment Distribution")
                fig = px.pie(df, names='Segment', title='Customer Segments Distribution')
                st.plotly_chart(fig)
                
                # Scatter plot
                st.subheader("Customer Segments Visualization")
                fig = px.scatter(df, 
                               x='Annual Income (k$)', 
                               y='Spending Score (1-100)',
                               color='Segment',
                               symbol='Gender',
                               title='Customer Segments by Income and Spending')
                st.plotly_chart(fig)
            else:
                st.error("Please upload a CSV file with the required columns: Gender, Age, Annual Income (k$), Spending Score (1-100)")

    # Add information about the segments
    st.markdown("---")
    st.subheader("About the Segments")
    segments_info = {
        "Affluent Enthusiasts": "High income and high spending customers who are prime targets for luxury products and premium services.",
        "Conservative High-Earners": "High income but low spending customers who might respond to value-proposition marketing.",
        "Budget Shoppers": "Lower income but high spending customers who might benefit from budget-friendly payment plans.",
        "Value-Conscious": "Lower income and low spending customers who are price-sensitive and look for deals.",
        "Average Consumers": "Middle-range customers in both income and spending who respond to balanced offerings."
    }
    
    for segment, description in segments_info.items():
        st.markdown(f"**{segment}**: {description}")

except FileNotFoundError:
    st.error("Model files not found. Please ensure the model is trained and saved in the 'models' directory.")