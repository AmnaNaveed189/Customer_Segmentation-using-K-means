# Customer Segmentation App

This Streamlit app demonstrates customer segmentation using K-means clustering.

## Setup Instructions

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Features

- Single customer segment prediction
- Batch prediction through CSV upload
- Interactive visualizations
- Detailed segment descriptions
- Download functionality for batch predictions

## Required CSV Format

For batch predictions, upload a CSV file with these columns:
- Gender (Female/Male)
- Age (numeric)
- Annual Income (k$) (numeric)
- Spending Score (1-100) (numeric)

## Model Details

The app uses a trained K-means clustering model with:
- Standardized features
- Categorical encoding for gender
- Optimal number of clusters determined by silhouette score
- Meaningful segment labels based on customer characteristics
