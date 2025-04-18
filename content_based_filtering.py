import streamlit as st
import pandas as pd
import pickle
import numpy as np
import scipy.sparse
import base64
import math
# Get user's language choice
language = st.session_state.get("language", "English")

def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

image_base64 = get_base64_image("example.jpg")
st.markdown("""
            <link href="https://fonts.googleapis.com/css2?family=Baloo+Chettan+2&display=swap" rel="stylesheet">
            <style>
                .baloo {
                    font-family: 'Baloo Chettan 2', cursive;
                }
                .helvetica {
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                }
            </style>
        """, unsafe_allow_html=True)
# === 📊 Dataset Setup ===
dataset_name_user_rating = "Products_ThoiTrangNam_rating.csv"
dataset_name_products = "Products_ThoiTrangNam_downsize.csv"

@st.cache_data
def load_data():
    user_rating = pd.read_csv(dataset_name_user_rating)
    products = pd.read_csv(dataset_name_products)
    products =products.drop(columns=['Unnamed: 0'])

    return user_rating, products

user_rating_df, products_df = load_data()

tfidf_matrix = scipy.sparse.load_npz('tfidf_matrix.npz')

with open('gensim.pkl', 'rb') as f:
    model2 = pickle.load(f)

def content_based_filtering(product_id, top_n=5):
    try:
        idx = products_df.index[products_df['product_id'] == product_id][0]
    except IndexError:
        return f"❌ Product ID '{product_id}' not found in dataset."

    # Get top N similar product indices and distances
    distances, indices = model2.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)  # +1 to skip self
    # Drop the first index (which is the product itself)
    similar_indices = indices.flatten()[1:]
    # Fetch recommended product info
    valid_indices = [i for i in similar_indices if i < len(products_df)]

    # Safely fetch recommended product info
    recommendation = products_df.iloc[valid_indices].copy()
    recommendation =recommendation.rename(columns={"rating": "EstimateScore"})
    
    recommendation = recommendation.sort_values(by='EstimateScore', ascending=False)
    return recommendation

custom_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px">
        <span style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips" if language == "English" else "Gợi ý"}:</strong> {"Recommendations will be generated based on the selected product, providing similar items." if language == "English" else "Dựa trên sản phẩm bạn đã chọn, hệ thống sẽ gợi ý những sản phẩm tương tự hoặc liên quan."}
        </span>
    </div>
    """

recommend_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px">
        <div style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips 1" if language == "English" else "Gợi ý 1"}:</strong> {"Please select the number of products to recommend (e.g., 5, 10, 15... products)" if language == "English" else "Hãy lựa chọn số lượng sản phẩm muốn gợi ý ( vd: 5,10,15...sản phẩm)"}
        </div>
        <div style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips 2" if language == "English" else "Gợi ý 2"}:</strong> {"You can receive recommendation results by specific product category or across all categories." if language == "English" else "Bạn có thể nhận kết quả gợi ý theo từng danh mục ngành hàng hoặc tất cả ngành hàng."}
        </div>
    </div>
    """
    
product_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px; margin-top: 20px">
        <div style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips 1" if language == "English" else "Gợi ý 1"}:</strong> {"Choose by product name or product ID." if language == "English" else "Lựa chọn theo tên sản phẩm hoặc ID sản phẩm."}
        </div>
        <div style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips 2" if language == "English" else "Gợi ý 2"}:</strong> {"Select the number of products to recommend." if language == "English" else "Chọn số lượng sản phẩm muốn gợi ý"}
        </div>
        <div style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips 3" if language == "English" else "Gợi ý 3"}:</strong> {"Select a specific or all categories from which you want to receive recommendations." if language == "English" else "Chọn danh mục từng ngành hàng hoặc tất cả ngành hàng mà bạn muốn nhận kết quả"}
        </div>
    </div>
    """
    
st.markdown("### 🧠 Recommendation using Content-based Filtering")

if language == "English":
        st.info("Content-Based Filtering recommends products based on similarities in product features.")
else:
        st.info("Lọc theo nội dung sẽ gợi ý sản phẩm dựa trên đặc điểm của sản phẩm.")
        
st.markdown("---")
product_names = products_df['product_name'].dropna().unique()
top_users = (
    user_rating_df['user']
    .dropna()
    .loc[~user_rating_df['user'].str.contains(r'\*', regex=True)]
    .value_counts()
    .head(100)
    .index
)
users_name = top_users.tolist()
st.markdown(
    f"""
    <div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
        👤 {"Choose a product to receive related product recommendations:" if language == "English"
        else "Chọn một sản phẩm để nhận các gợi ý sản phẩm liên quan."}
    </div>
    """, unsafe_allow_html=True)
    
st.markdown(product_info, unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])  # Equal width columns

with col1:
        selection_mode = st.selectbox(
            label="Name or Product ID" if language == "English" else "Tên hoặc mã sản phẩm",
            options=("👤 Product Name", "🆔 Product ID") if language == "English" 
                    else ("👤 Tên sản phẩm", "🆔 Mã sản phẩm"),
            index = 0
        )

with col2:
        num_products1 = st.selectbox(
            "Number of products to display" if language == "English" else "Số lượng sản phẩm hiển thị",
            options=[5, 10, 20, 50],
            index=2 
        )

with col3:
        # === 📂 Get Unique Categories for Filtering ===
        categories = products_df["sub_category"].dropna().unique().tolist()
        selected_category = st.selectbox(
            "Filter by Category" if language == "English" else "Lọc theo chủng loại sản phẩm",
            options=["All"] + categories,
            index=0,
            key="category_select"
        )
if (selected_category != "All"):
        products_df2 = products_df[products_df["sub_category"] == selected_category]
else:
        products_df2 = products_df
if "Name" in selection_mode or "Tên" in selection_mode:
        top_products = (
            products_df2['product_name']
            .dropna()
            .loc[~products_df['product_name'].str.contains(r'\*', regex=True)]
            .value_counts()
            .index
        )
else:
        top_products = (
            products_df2['product_id']
            .dropna()
            .value_counts()
            .index
        )

product_names = top_products.tolist()
selected_product = st.selectbox(
        label="",
        label_visibility = 'hidden',
        options=product_names,
        index=0,
        placeholder="Type to search..." if language == "English" else "Nhập để tìm..."
    )
if "Name" in selection_mode or "Tên" in selection_mode:
    # Map the selected name to the corresponding user_id(s) — take the most active one if duplicates
        product_id = (
            products_df[products_df['product_name'] == selected_product]
            .groupby('product_id')
            .size()
            .sort_values(ascending=False)
            .index[0]
        )
else:
        product_id = selected_product
st.markdown("---")

button_clicked = st.button("🔍 Generate Recommendations" if language == "English" else "🔍 Nhận các sản phẩm gợi ý", use_container_width=True)
if button_clicked:
    recommendations = content_based_filtering(product_id, num_products1)

    suggested_products = []
    for _, row in recommendations.iterrows():
            suggested_products.append({
                "name": row["product_name"],
                "image_url": row.get("image") or "shopee.jpg",
                "product_url": row.get("link", "https://example.com/product"),  # default if missing
                "product_category": row["sub_category"],
                "price": int(row.get("price", 0)),
                "rating": float(row.get("EstimateScore", 0)),
                # "description": row.get("description_clean", "No description available.")
            })
    product_name = products_df.loc[products_df["product_id"] == product_id, "product_name"].values
    content = product_name[0]
    print(content)
    what = "based on" if language == "English" else "dựa trên"
    st.markdown(
        f"### 🛒 Recommended Products {what} - {content}"
        if language == "English"
        else f"### 🛒 Sản phẩm gợi ý {what} - {content}"
    )

    st.markdown(
    """
    <style>
    .scrolling-wrapper {
        overflow-x: auto;
        display: flex;
        flex-wrap: nowrap;
        padding-bottom: 10px;
        gap: 16px;
        scroll-snap-type: x mandatory;
    }
    .card {
        flex: 0 0 auto;
        scroll-snap-align: start;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        width: 300px;
    }
    .card img {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .product-meta {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 0.3rem;
        color:black;
    }
    .product-name {
        font-weight: 300;
        font-size: 16px;
        height: 80px;
        overflow: hidden;
        color: black;
    }
    .price {
        font-weight: 800;
        color: black;
    }
    .rating {
        font-weight: 1000;
        color: black;
        margin-bottom: 0.5rem;
    }
    .btn {
        padding: 0.5rem 1rem;
        background-color: black;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True)

    scrollable_html = '<div class="scrolling-wrapper">'

    for product in suggested_products:
        image_url = product.get("image_url")
        if not image_url or (isinstance(image_url, float) and math.isnan(image_url)):
            image_url = f"data:image/jpg;base64,{image_base64}"
        scrollable_html += f"""<div class="card">
            <img src="{image_url}" alt="{product['name']}"/>
            <div class="product-meta">
                <div>MEN</div>
                <div>{product['product_category']}</div>
            </div>
            <div class="product-name">{product['name']}</div>
            <div class="price">{product['price']} VND</div>
            <div class="rating">⭐ {product['rating']}</div>
            <a href="{product['product_url']}" target="_blank">
                <button class="btn">View Product</button>
            </a>
        </div>
    """

    scrollable_html += '</div>'

    st.markdown(scrollable_html, unsafe_allow_html=True)