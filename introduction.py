import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

# Load the image
image = Image.open('shopee.jpg')

# Display the image
st.write("""# Data Science Project - Shopee""")
st.image(image, caption='Shopee Recommendation System', use_column_width=True)


st.write("""
---
#### 📦 Project Title: 
- ***Shopee*** Product Recommendation System

#### 🧠 Project Description:
- This project focuses on building a personalized recommendation system for Shopee, a popular e-commerce platform. The goal is to enhance user experience by suggesting relevant products based on user behavior, product metadata, and historical transaction data.

- Using data science techniques such as collaborative filtering, content-based filtering, and hybrid models, the system predicts what users are likely to purchase next. This can significantly increase user engagement, conversion rates, and overall customer satisfaction.

#### ⚙️ Key Features:
- Personalized product recommendations for each user.

#### 📊 Tech Stack & Tools:
- `Languages`: Python

- `Libraries`: Pandas, NumPy, Scikit-learn, SurPRISE, Underthesea, Cosine Similarity

- `Visualization`: Matplotlib, Seaborn

- `Deployment`: Streamlit

#### 📈 Outcomes:
- Improved recommendation precision and recall.

- Clear insights into customer behavior.

- Ready-to-integrate API or demo showcasing recommendation functionality.

***
""")
