import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

# Load the image
image = Image.open('shopee.jpg')

# Language selector (or get from session state if set globally)
language = st.session_state.get("language", "English")

# Display title and image
st.write("# Data Science Project - Shopee")
st.image(image, caption='Shopee Recommendation System', use_container_width=True)

# Content for English
if language == "English":
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

# Content for Vietnamese
else:
    st.write("""
    ---
    #### 📦 Tên Dự Án: 
    - Hệ thống Gợi ý Sản phẩm ***Shopee***

    #### 🧠 Mô tả Dự Án:
    - Dự án này tập trung vào việc xây dựng hệ thống gợi ý cá nhân hóa cho Shopee – một nền tảng thương mại điện tử phổ biến. Mục tiêu là nâng cao trải nghiệm người dùng bằng cách đề xuất các sản phẩm phù hợp dựa trên hành vi, thông tin sản phẩm và dữ liệu giao dịch lịch sử.

    - Sử dụng các kỹ thuật khoa học dữ liệu như collaborative filtering, content-based filtering và mô hình kết hợp, hệ thống dự đoán những sản phẩm mà người dùng có thể quan tâm hoặc sẽ mua tiếp theo. Điều này có thể tăng cường sự tương tác, tỷ lệ chuyển đổi và sự hài lòng của khách hàng.

    #### ⚙️ Tính Năng Chính:
    - Gợi ý sản phẩm cá nhân hóa cho từng người dùng.

    #### 📊 Công Nghệ & Công Cụ:
    - `Ngôn ngữ`: Python

    - `Thư viện`: Pandas, NumPy, Scikit-learn, SurPRISE, Underthesea, Cosine Similarity

    - `Trực quan hóa`: Matplotlib, Seaborn

    - `Triển khai`: Streamlit

    #### 📈 Kết Quả:
    - Tăng độ chính xác và độ bao phủ trong gợi ý.

    - Hiểu rõ hơn về hành vi khách hàng.

    - Tích hợp dễ dàng qua API hoặc bản demo.

    ***
    """)
