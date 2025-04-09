import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image


# Language selector
with open("user_guide.svg", "r") as f:
    svg_content = f.read()

# OR better SVG rendering:
st.markdown(f"<div>{svg_content}</div>", unsafe_allow_html=True)
language = st.session_state.get("language", "English")

if language == "English":
    st.write("""
    # 🧑‍💻 User Guide

    Welcome to the **Shopee Product Recommendation System**! This guide will walk you through using the system, understanding the models, and exploring the dataset.

    ---

    ## ⚙️ Model Preparation

    The system supports two recommendation techniques:

    ### 🔹 Content-Based Filtering
    Recommends similar items based on product content (e.g., name, description, category).

    - `cosine.pkl`: Uses cosine similarity between product vectors.
    - `gensim.pkl`: Uses word embeddings (Word2Vec / Doc2Vec) trained with Gensim.

    ### 🔹 Collaborative Filtering
    Recommends items based on user behavior and preferences.

    - `als.pkl`: Matrix factorization using the ALS algorithm.
    - `surprise.pkl`: Collaborative filtering model using the Surprise library.

    ---

    ## 📦 Datasets

    ### 📁 Project 1 – General Product Recommendation
    - `Products_with_Categories.csv`: Product metadata (name, category).
    - `Transactions.csv`: User-product interaction history.

    ### 📁 Project 2 – Fashion (Men's Clothing)
    - `Products_ThoiTrangNam.csv`: Product metadata for men’s fashion.
    - `Products_ThoiTrangNam_rating.csv`: User ratings for fashion products.

    You can switch between datasets depending on your focus.

    ---

    ## 🧭 Navigation

    The project is structured across several main pages:

    ### 🏠 Introduction
    Overview of the project, goals, and recommendation approaches.

    ### 📘 User Guide
    Detailed instructions for using the app and understanding the models.

    ### 📊 EDA – Exploratory Data Analysis
    Visual exploration of the dataset:
    - Popular product categories
    - Top purchased products
    - User activity patterns

    ### 🤖 Recommendation
    Try out the recommender system:
    - Select product or user
    - Choose model (Content-based or Collaborative)
    - Get top-N product recommendations

    ---

    ## 🔗 GitHub Repository

    Access the full code, models, and documentation here:  
    👉 [GitHub Repository](https://github.com/anatwork14/data_science_recommend_system.git)

    ---

    Enjoy exploring and recommending with Shopee Recommendation System! 🚀
    """)

elif language == "Tiếng Việt":
    st.write("""
    # 🧑‍💻 Hướng Dẫn Sử Dụng

    Chào mừng bạn đến với **Hệ thống Gợi ý Sản phẩm Shopee**! Hướng dẫn này sẽ giúp bạn sử dụng hệ thống, hiểu các mô hình gợi ý và khám phá tập dữ liệu.

    ---

    ## ⚙️ Chuẩn Bị Mô Hình

    Hệ thống hỗ trợ hai kỹ thuật gợi ý:

    ### 🔹 Gợi ý dựa trên nội dung
    Đề xuất sản phẩm tương tự dựa trên nội dung (ví dụ: tên, mô tả, danh mục).

    - `cosine.pkl`: Sử dụng độ tương đồng cosine giữa các vector sản phẩm.
    - `gensim.pkl`: Sử dụng word embeddings (Word2Vec / Doc2Vec) huấn luyện bằng Gensim.

    ### 🔹 Gợi ý dựa trên cộng tác
    Đề xuất sản phẩm dựa trên hành vi và sở thích người dùng.

    - `als.pkl`: Phân rã ma trận sử dụng thuật toán ALS.
    - `surprise.pkl`: Mô hình cộng tác sử dụng thư viện Surprise.

    ---

    ## 📦 Tập Dữ Liệu

    ### 📁 Dự Án 1 – Gợi ý sản phẩm chung
    - `Products_with_Categories.csv`: Thông tin sản phẩm (tên, danh mục).
    - `Transactions.csv`: Lịch sử tương tác giữa người dùng và sản phẩm.

    ### 📁 Dự Án 2 – Thời trang nam
    - `Products_ThoiTrangNam.csv`: Thông tin sản phẩm thời trang nam.
    - `Products_ThoiTrangNam_rating.csv`: Đánh giá sản phẩm của người dùng.

    Bạn có thể chuyển đổi giữa các tập dữ liệu tùy theo mục đích.

    ---

    ## 🧭 Điều Hướng

    Dự án được chia thành các trang chính:

    ### 🏠 Giới thiệu
    Tổng quan dự án, mục tiêu và các phương pháp gợi ý.

    ### 📘 Hướng dẫn sử dụng
    Hướng dẫn chi tiết cách sử dụng ứng dụng và hiểu các mô hình.

    ### 📊 Phân tích dữ liệu (EDA)
    Khám phá dữ liệu qua biểu đồ:
    - Các danh mục phổ biến
    - Sản phẩm được mua nhiều nhất
    - Hành vi người dùng

    ### 🤖 Gợi ý sản phẩm
    Trải nghiệm hệ thống gợi ý:
    - Chọn sản phẩm hoặc người dùng
    - Chọn mô hình (nội dung hoặc cộng tác)
    - Nhận top-N sản phẩm gợi ý

    ---

    ## 🔗 Kho Mã Nguồn GitHub

    Truy cập mã nguồn, mô hình và tài liệu tại:  
    👉 [GitHub Repository](https://github.com/anatwork14/data_science_recommend_system.git)

    ---

    Chúc bạn trải nghiệm vui vẻ cùng Shopee Recommendation System! 🚀
    """)
