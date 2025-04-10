import streamlit as st
import pandas as pd
import pickle
import numpy as np
import scipy.sparse

# Get user's language choice
language = st.session_state.get("language", "English")

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

# === Model set up ===)
with open('surprise.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the model

def collab_filtering(user_id, top_n=5, categories = 'All'):
    # Filter products rated >= 3 by this user (historical data, not used in prediction)
    user_history = user_rating_df[(user_rating_df['user_id'] == user_id) & (user_rating_df['rating'] >= 3)]
    
    # Get all unique product IDs from products_df
    candidate_products = products_df[['product_id']].drop_duplicates()

    # Predict estimated ratings for each product
    candidate_products['EstimateScore'] = candidate_products['product_id'].apply(
        lambda pid: model.predict(user_id, pid).est
    )

    # Remove products the user has already rated
    candidate_products = candidate_products[~candidate_products['product_id'].isin(user_history['product_id'])]

    # Join with product info
    temp = products_df.drop(columns="rating")
    recommendations = candidate_products.merge(temp, on='product_id', how='left')

    # Drop products with missing names (if any)
    recommendations = recommendations.dropna(subset=['product_name'])
    
    if (categories != "All"):
        recommendations = recommendations[recommendations["sub_category"] == categories]
    # Sort by predicted score
    recommendations["EstimateScore"] = recommendations["EstimateScore"].round(2)

    recommendations = recommendations.sort_values(by='EstimateScore', ascending=False)

    if (len(recommendations) < top_n):
        return recommendations.head(len(recommendations))
    return recommendations.head(top_n)

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


# === 🎯 Filtering Method Selection ===
st.markdown("## 🎯 Recommendation Filtering Method" if language == "English" else "## 🎯 Phương pháp gợi ý")

filtering_method = st.selectbox(
    "🔍 Choose your recommendation approach:" if language == "English"
    else "🔍 Chọn phương pháp gợi ý bạn muốn sử dụng:",
    ("🧠 Content-Based Filtering", "🤝 Collaborative Filtering") if language == "English"
    else ("🧠 Lọc theo nội dung", "🤝 Lọc cộng tác")
)

# Display a description below the selection
if language == "English":
    if "Content" in filtering_method:
        st.info("Content-Based Filtering recommends products based on similarities in product features.")
    else:
        st.info("Collaborative Filtering recommends products based on user behavior and preferences.")
else:
    if "nội dung" in filtering_method:
        st.info("Lọc theo nội dung sẽ gợi ý sản phẩm dựa trên đặc điểm của sản phẩm.")
    else:
        st.info("Lọc cộng tác sẽ gợi ý sản phẩm dựa trên hành vi và sở thích của người dùng.")

# Horizontal rule
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

if filtering_method == "🧠 Content-Based Filtering" or filtering_method == "🧠 Lọc theo nội dung":
    custom_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px">
        <span style="color: #0a66c2; font-size: 16px;">
            <strong>📌 {"Tips" if language == "English" else "Gợi ý"}:</strong> {"Recommendations will be generated based on the selected product, providing similar items." if language == "English" else "Dựa trên sản phẩm bạn đã chọn, hệ thống sẽ gợi ý những sản phẩm tương tự hoặc liên quan."}
        </span>
    </div>
    """
    st.markdown(custom_info, unsafe_allow_html=True)
    
    st.markdown(
    f"""
    <div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
        👤 {"Select a user to get personalized recommendations:" if language == "English"
        else "Chọn người dùng để nhận gợi ý cá nhân hóa:"}
    </div>
    """,
    unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])  # Equal width columns

    with col1:
        selection_mode = st.radio(
            label="",
            label_visibility='hidden',
            options=("👤 Product Name", "🆔 Product ID") if language == "English" 
                    else ("👤 Tên sản phẩm", "🆔 Mã sản phẩm"),
            horizontal=True
        )

    with col2:
        num_products1 = st.selectbox(
            "Number of products to display" if language == "English" else "Số lượng sản phẩm hiển thị",
            options=[5, 10, 20, 50],
            index=1  # default to 10
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
else:
    custom_info = f"""
    <div style="background-color: #e8f4fd; padding: 10px 12px; border-radius: 16px; margin-bottom: 20px">
        <span style="color: #0a66c2; font-size: 16px;">
             <strong>📌 {"Tips" if language == "English" else "Gợi ý"}:</strong> 
            {"Recommendations will be based on the selected user, suggesting products similar to those they have previously engaged with." 
            if language == "English" else "Dựa trên người dùng bạn đã chọn, hệ thống sẽ gợi ý các sản phẩm tương tự với những gì họ đã từng mua hoặc tương tác."}
        </span>
    </div>
    """
    st.markdown(custom_info, unsafe_allow_html=True)
    # 👇 Language-aware label
    st.markdown(
    f"""
    <div style='font-size: 1.25rem; font-weight: 600; margin-bottom: 0rem;'>
        👤 {"Select a user to get personalized recommendations:" if language == "English"
        else "Chọn người dùng để nhận gợi ý cá nhân hóa:"}
    </div>
    """,
    unsafe_allow_html=True)

    # 👇 Then use radio without label
    
    col1, col2, col3 = st.columns([1, 1, 1])  # Equal width columns

    with col1:
        selection_mode = st.radio(
            label="",
            label_visibility='hidden',
            options=("👤 User Name", "🆔 User ID") if language == "English" 
                    else ("👤 Tên người dùng", "🆔 Mã người dùng"),
            horizontal=True
        )

    with col2:
        num_products2 = st.selectbox(
            "Number of products to display" if language == "English" else "Số lượng sản phẩm hiển thị",
            options=[5, 10, 20, 50],
            index=1  # default to 10
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

    selected_user = st.selectbox(
        label="",
        label_visibility = 'hidden',
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
    st.markdown("---")

recommendations = content_based_filtering(product_id, num_products1) if (filtering_method == "🧠 Content-Based Filtering" or filtering_method == "🧠 Lọc theo nội dung") else collab_filtering(user_id, num_products2, selected_category)

suggested_products = []
for _, row in recommendations.iterrows():
        suggested_products.append({
            "name": row["product_name"],
            "image_url": row.get("image") or "shopee.jpg",
            "product_url": row.get("link", "https://example.com/product"),  # default if missing
            "product_category": row["sub_category"],
            "price": int(row.get("price", 0)),
            "rating": float(row.get("EstimateScore", 0)),
            "description": row.get("description_clean", "No description available.")
        })
content = selected_product if (filtering_method == "🧠 Content-Based Filtering" or filtering_method == "🧠 Lọc theo nội dung") else selected_user

what = ("based on" if language == "English" else "dựa trên") if (filtering_method == "🧠 Content-Based Filtering" or filtering_method == "🧠 Lọc theo nội dung") else ("for User" if language == "English" else "dựa trên")
if (len(suggested_products) > 0):
    # === 🌟 Section Title ===
        st.markdown(f"### 🛒 Recommended Products {what} - {content}" if language == "English" else f"### 🛒 Sản phẩm gợi ý {what} - {content}")

        # === 📄 Pagination Controls ===
        items_per_page = 3
        total_pages = -(-len(suggested_products) // items_per_page)  # ceiling division
        
        page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=0,
            key="pagination_select"
        )

        # === 🔄 Paginate Items ===
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paged_products = suggested_products[start_idx:end_idx]

        # === 🎨 Display in Columns ===
        cols = st.columns(len(paged_products))

        # Import Google Font only once (place this near the top of your app)
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

        # Product display with Baloo Chettan 2
        for i, product in enumerate(paged_products):
            with cols[i]:
                import base64
                import math
                def get_base64_image(image_path):
                    with open(image_path, "rb") as f:
                        return base64.b64encode(f.read()).decode()

                image_url = product.get("image_url")
                if not image_url or (isinstance(image_url, float) and math.isnan(image_url)):
                    image_base64 = get_base64_image("example.jpg")
                    image_url = f"data:image/jpg;base64,{image_base64}"
                description = product.get("description")
                if not description or (isinstance(description, float) and math.isnan(description)):
                    description = "No description" if language == "English" else "Không có mô tả"
                st.markdown(
                    f"""
                    <div style='
                        background-color: #ffffff;
                        border: 1px solid #e0e0e0;
                        border-radius: 12px;
                        padding: 1.5rem;
                        margin-bottom: 2rem;
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
                        font-size: 1.1rem;
                        font-family: "Baloo Chettan 2", cursive;
                        text-align: left;
                        color: #333333;
                        transition: box-shadow 0.3s ease, transform 0.2s ease;
                    '>
                        <img src="{image_url}" alt="{product['name']}" style="width: 100%; border-radius: 8px; margin-bottom: 0.8rem;" />
                        <div class ="helvetica" style="display: flex; justify-content: space-between; align-items: center; font-size: 13px">
                            <div style="font-weight: 200; color: black">MEN</div>
                            <div style="font-weight: 100; font-size: 12px; color: black;">{product["product_category"]}</div>
                        </div>
                        <div class="helvetica" style='font-weight: 300; font-size: 16px; color: black; height: 80px; overflow: hidden;'>{product['name']}</div>
                        <div class="helvetica" style='color: black; font-weight: 800;'>{product['price']} VND</div>
                        <div style='color: black; font-weight: 1000'>⭐ {product['rating']}</div>
                        <a href="{product['product_url']}" target="_blank">
                            <button style='
                                margin-top: 0.8rem;
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
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("Product Description"):
                        st.markdown(f"""
                            <div style='
                                font-size: 12px;
                                color: white;
                                width: 100%;
                                padding: 8px 4px;
                                word-wrap: break-word;
                                white-space: normal;  
                            '>
                                {description}
                            </div>
                        """, unsafe_allow_html=True)
else: 
        st.markdown(f"### ❌ Không tìm thấy sản phẩm gợi ý | người dùng")
# 🧠 Example product suggestions (replace this with your own logic)
