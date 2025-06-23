import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import LabelEncoder

import base64
# Function to load and encode local jpg image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Local image filename (same folder)
image_file = 'asa.jpg'

# Get base64 string
img_base64 = get_base64_of_bin_file(image_file)

# Inject HTML + CSS for background
page_bg_img = f"""
<style>
.stApp {{
  background-image: url("data:image/jpg;base64,{img_base64}");
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}
</style>
"""

# Load CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Introduction", "Clustering", "Recommendation", "Creator Info"])


# -------------------------------- PAGE 1: Introduction --------------------------------
if page == "Project Introduction":
    st.title("🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce ")
    st.write("\n")
    st.write("\n")
    st.write(""" 
    ##### The global e-commerce industry generates vast amounts of transaction data daily, offering 
##### valuable insights into customer purchasing behaviors. Analyzing this data is essential for 
##### identifying meaningful customer segments and recommending relevant products to enhance 
##### customer experience and drive business growth. """)
    st.write("\n")
    st.write("\n")

    st.markdown("""\n
    ### Real-time Business Use Cases:  \n
        ● Customer Segmentation for Targeted Marketing Campaigns \n
        ● Personalized Product Recommendations on E-Commerce Platforms \n
        ● Identifying At-Risk Customers for Retention Programs \n
        ● Dynamic Pricing Strategies Based on Purchase Behavior \n
        ● Inventory Management and Stock Optimization Based on Customer Demand Patterns  """)

    st.markdown("""
     ### Problem Type: \n
        ● Unsupervised Machine Learning – Clustering \n
        ● Collaborative Filtering – Recommendation System """)
        
    #st.image(r'C:\Users\91968\OneDrive\Desktop\Pthon DS GuVi\Project\Project2\images (11).jpeg', width=152)

# -------------------------------- PAGE 2: Clustering --------------------------------
elif page == "Clustering":
    st.title("🧠 Customer Segmentation")
    st.write("\n")

    # Load the scaler used for RFM normalization
    with open("rfm_scaled.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model.pkl", "rb") as e:
        kmeans = pickle.load(e)

    # Define your segment labels
    segment_labels = {
        0: "At-Risk",
        1: "Occational",
        2: "High-Value",
        3: "High Value",
        4: "Regular"
    }

    st.markdown("""
    Enter **Recency**, **Frequency**, and **Monetary** values to predict the customer segment.
    """)
    st.write("\n")
    c1,c2,c3 = st.columns([1,1,1], gap = 'large')

    # Input fields
    with c1 :
        recency = st.number_input("**Recency (days since last purchase)**", min_value=0)
    with c2 :
        frequency = st.number_input("**Frequency (number of purchases)**", min_value=0)
    with c3 :
        monetary = st.number_input("**Monetary (total spending)**", min_value=0.0, step=10.0)

    # Prediction
    if st.button("Predict Segment"):
        # Ensure input is in correct 2D format
        rfm1 = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm1)
        cluster = kmeans.predict(rfm_scaled)[0]
        segment = segment_labels.get(cluster, "Unknown")

        st.success(f"Cluster is : **{cluster}**")

        st.success(f"The predicted customer segment is :  **{segment}**")

# -------------------------------- PAGE 3: Recommendation --------------------------------

elif page == "Recommendation":
    st.title("📋 Product Recommendation")
    data = pd.read_csv('data_clean.csv')


    # Create a mapping dictionary
    stockcode_to_name = data[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Example: customer-item matrix (binary or count-based)
    item_matrix = data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum', fill_value=0)
    
    # Transpose for item-to-item similarity
    item_sim = cosine_similarity(item_matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=item_matrix.columns, columns=item_matrix.columns)
    
    def get_similar_products(stock_code, top_n=5):
        similar_codes = item_sim_df[stock_code].sort_values(ascending=False).iloc[1:top_n+1]
        result = [(code, stockcode_to_name.get(code, 'Unknown')) for code in similar_codes.index]
        return result
        
    st.markdown("Enter a **Product StockCode** to get top 5 similar products (based on customer purchase history).")

    stock_code_input = st.text_input("Enter StockCode (e.g., 85123A) : ")
   
    if st.button("Get Recommendations"):
        if stock_code_input.strip() == "":
            st.warning("Please enter a valid StockCode.")
        elif stock_code_input not in item_sim_df.index:
            st.error("StockCode not found in data.")
        else:
            top_n = 5
            similar_products = get_similar_products(stock_code_input, top_n=top_n)

            st.markdown(f"### 🔎 Top {top_n} recommendations for StockCode: `{stock_code_input}`")
            for i, (code, name) in enumerate(similar_products, 1):
                st.write(f"{i}. { name } (StockCode: `{code}`)")
                
# -------------------------------- PAGE 4: Creator Info --------------------------------

elif page == "Creator Info":
    st.title("👨‍💻 Creator of this Project")
    st.write("""
#    **Developed by:** Sudharsan M S
#    **Skills:** 
## Python,   
## Machine Learning,   
## Streamlit
    """)
    st.image('aa.jpg', width=150)


  
