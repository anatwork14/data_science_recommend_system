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

# === Model set up ===
with open('surprise.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the model
def collab_filtering(user_id, top_n=5, categories='All'):
    # Get user history of positively rated items
    user_history = user_rating_df[
        (user_rating_df['user_id'] == user_id) & 
        (user_rating_df['rating'] >= 3)
    ]

    # Get all unique product IDs
    all_product_ids = products_df['product_id'].unique()

    # Keep only products known to the model
    known_items = set(model.trainset._raw2inner_id_items.keys())
    known_product_ids = [pid for pid in all_product_ids if pid in known_items]

    # Predict scores for known items
    predictions = [
        (pid, model.predict(user_id, pid).est) 
        for pid in known_product_ids 
        if pid not in user_history['product_id'].values
    ]

    # Build DataFrame of predictions
    pred_df = pd.DataFrame(predictions, columns=['product_id', 'EstimateScore'])

    # Merge with product info
    recommendations = pred_df.merge(
        products_df.drop(columns='rating', errors='ignore').drop_duplicates(subset='product_id'),
        on='product_id',
        how='left'
    ).dropna(subset=['product_name'])

    # Filter by category
    if categories != "All":
        recommendations = recommendations[recommendations["sub_category"] == categories]

    # Round and sort
    recommendations["EstimateScore"] = recommendations["EstimateScore"].round(2)
    recommendations = recommendations.sort_values(by='EstimateScore', ascending=False)

    return recommendations.head(top_n if len(recommendations) >= top_n else len(recommendations))


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
    
st.markdown("### 🤝 Recommendation using Collaborative Filtering")
products = []

if language == "English":
        st.info("Collaborative Filtering recommends products based on user behavior and preferences.")
else:
        st.info("Lọc cộng tác sẽ gợi ý sản phẩm dựa trên hành vi và sở thích của người dùng.")
        
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

custom_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px; margin-top: 20px">
        <span style="color: #0a66c2; font-size: 16px;">
             <strong>📌 {"Tips" if language == "English" else "Gợi ý"}:</strong> 
            {"Recommendations will be based on the selected user, suggesting products similar to those they have previously engaged with." 
            if language == "English" else "Dựa trên người dùng bạn đã chọn, hệ thống sẽ gợi ý các sản phẩm tương tự với những gì họ đã từng mua hoặc tương tác."}
        </span>
    </div>
    """
    # st.markdown(custom_info, unsafe_allow_html=True)
    # 👇 Language-aware label
st.markdown(
    f"""
    <div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
        👤 {"Select a user to get personalized recommendations:" if language == "English"
        else "Chọn người dùng để nhận gợi ý cá nhân hóa:"}
    </div>
    """,
    unsafe_allow_html=True)
st.markdown(custom_info, unsafe_allow_html=True)

    # 👇 Then use radio without label
    
col1, col2 = st.columns([1, 1])  # Equal width columns

with col1:
        selection_mode = st.selectbox(
            label="Name or User ID" if language == "English" else "Tên hoặc Mã người dùng",
            options=("👤 User Name", "🆔 User ID") if language == "English" 
                    else ("👤 Tên người dùng", "🆔 Mã người dùng"),
            index=0
        )
# 👇 Load the top 100 users based on selection mode
if "Name" in selection_mode or "Tên" in selection_mode:
    top_users = (
            user_rating_df['user']
            .dropna()
            .loc[~user_rating_df['user'].str.contains(r'\*', regex=True)]
            .value_counts()
            .head(100)
            .index
        )
else:
    top_users = (
            user_rating_df['user_id']
            .dropna()
            .value_counts()
            .head(100)
            .index
    )

users_name = top_users.tolist()
with col2:
        selected_user = st.selectbox(
        label="User List" if language == "English" else "Danh sách người dùng",
        options=users_name,
        index=0,
        placeholder="Type to search..." if language == "English" else "Nhập để tìm..."
        )
    
if "Name" in selection_mode or "Tên" in selection_mode:
    # Map the selected name to the corresponding user_id(s) — take the most active one if duplicates
        user_id = (
            user_rating_df[user_rating_df['user'] == selected_user]
            .groupby('user_id')
            .size()
            .sort_values(ascending=False)
            .index[0]
        )
else:
    user_id = selected_user  # already a user_id
product_ids = (
    user_rating_df[user_rating_df["user_id"] == user_id]
    .drop_duplicates(keep="first")
    .sort_values("rating")  # Now sorting works because 'rating' is present
    .head(100)["product_id"]
    .tolist()
    )

for x in product_ids:
        product_row = products_df[products_df["product_id"] == x]
        
        if not product_row.empty:
            # Extract image
            image_url = product_row["image"].values[0]
            
            # Check for missing/NaN image
            if not image_url or (isinstance(image_url, float) and math.isnan(image_url)):
                image_url = f"data:image/jpg;base64,{image_base64}"

            products.append({
                "name": product_row["product_name"].values[0],
                "price": product_row["price"].values[0],
                "rating": product_row["rating"].values[0],
                "image": image_url,
                "category": product_row["sub_category"].values[0],
                "product_url": product_row["link"].values[0],
            })

    # CSS for horizontal scrolling
if language == "English":
        st.markdown(f"#### 🔄 Top {len(products)} products - {selected_user} has bought")
else:
        user_label = "Người dùng có mã" if "Name" in selection_mode or "Tên" not in selection_mode else ""
        st.markdown(f"#### 🔄 Top {len(products)} sản phẩm - {user_label} {selected_user} đã mua")

st.markdown("""
    <style>
    .product-container {
        display: flex;
        overflow-x: auto;
        padding: 1rem 0;
        gap: 1rem;
        scrollbar-width: thin;
        height: 478px
    }
    .product-card {
        flex: 0 0 auto;
        width: 240px;
        border: 1px solid #eaeaea;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .product-card img {
        width: 100%;
        height: auto;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Generate product cards
product_html = '<div class="product-container">'

for product in products:
        product_html += f"""<div class="product-card">
            <img src="{product['image']}" alt="{product['name']}"/>
            <div class ="helvetica" style="display: flex; justify-content: space-between; align-items: center; font-size: 13px">
                <div style="font-weight: 200; color: black">MEN</div>
                <div style="font-weight: 100; font-size: 12px; color: black;">{product["category"]}</div>
            </div>
            <div class="helvetica" style='font-weight: 300; font-size: 16px; color: black; height: 80px; overflow: hidden;'>{product['name']}</div>
            <div class="helvetica" style='color: black; font-weight: 800;'>{product['price']} VND</div>
            <div style='color: black; font-weight: 1000'>⭐ {product['rating']}</div>
            <a href="{product['product_url']}" target="_blank">
                            <button style='
                                margin-top: 0.8rem;
                                margin-bottom: 4px;
                                padding: 0.5rem 1rem;
                                background-color: black;
                                color: white;
                                border: none;
                                border-radius: 8px;
                                cursor: pointer;
                                font-family: "Baloo Chettan 2", cursive;
                                font-size: 1rem;
                                width: 100%
                            '>
                                View Product
                            </button>
            </a>
        </div>
        """

product_html += '</div>'


st.markdown(product_html, unsafe_allow_html=True)
st.markdown("---")

st.markdown(recommend_info, unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
        num_products2 = st.selectbox(
                    "Number of products to suggest" if language == "English" else "Số lượng sản phẩm gợi ý",
                    options=[5, 10, 20, 50],
                    index=2  # default to 10
                )

        # === 📂 Get Unique Categories for Filtering ===
categories = products_df["sub_category"].dropna().unique().tolist()

with col2: 
        selected_category = st.selectbox(
                    "Filter by Category" if language == "English" else "Lọc theo chủng loại sản phẩm",
                    options=["All"] + categories,
                    index=0,
                    key="category_select"
                )
        
button_clicked = st.button("🔍 Generate Recommendations" if language == "English" else "🔍 Nhận các sản phẩm gợi ý", use_container_width=True)

# === Step 1: Trigger recommendation only on button click ===
if button_clicked:
    recommendations = collab_filtering(user_id, num_products2, selected_category)

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
            
    product_name = user_rating_df.loc[user_rating_df["user_id"] == user_id, "user"].values
    content = product_name[0]

    what = "for User" if language == "English" else "dành cho"
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