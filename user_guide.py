import streamlit as st
import numpy as np
import pandas as pd
import time

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
👉 [GitHub Repository](https://github.com/your-repo)

---

Enjoy exploring and recommending with Shopee AI! 🚀""")