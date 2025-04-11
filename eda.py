import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
import plotly.express as px

# Set language from session
language = st.session_state.get("language", "English")

dataset_name_user_rating = "Products_ThoiTrangNam_rating.csv"
dataset_name_products = "Products_ThoiTrangNam_downsize.csv"
# === 🧹 Load the Data ===
@st.cache_data
def load_data():
    user_rating = pd.read_csv(dataset_name_user_rating)
    products = pd.read_csv(dataset_name_products)
    products =products.drop(columns=['Unnamed: 0'])
    overall = pd.merge(user_rating, products, on='product_id', how = 'left')
    overall = overall.rename(columns={'rating_x': 'user_rating', 'rating_y': 'product_overall_rating'})
    
    
    return user_rating, products, overall

user_rating_df, products_df, overall_df = load_data()

selected_dataset = st.sidebar.selectbox(
    "Choose a dataset to analyze:" if language == "English" else "Chọn dữ liệu để phân tích:",
    ("Products Ratings", "User Ratings")
)

st.session_state.selected_dataset = selected_dataset
dataset = []

if (selected_dataset == "Products Ratings"):
    dataset = products_df
elif (selected_dataset == "User Ratings"): 
    dataset = user_rating_df
else:
    dataset = overall_df
text = ""
# === 🧾 Dataset Summary ===
if language == "English":
    st.subheader(f"📊 {selected_dataset} Dataset Summary")
    st.write("Shape:", dataset.shape)
    st.write("Sample Data:")
    text = "Select the number of samples to display"

else:
    st.subheader(f"📊 Thống kê Dữ liệu cho {selected_dataset}")
    st.write("Kích thước:", dataset.shape)
    st.write("Dữ liệu mẫu:")
    text = "Chọn số lượng mẫu dữ liệu để hiển thị"
    
value = value = st.slider(text, min_value=0, max_value=100, value=5)
st.dataframe(dataset.head(value))
if (selected_dataset != "Overall for both"):
    if (selected_dataset == "Products Ratings"):
        # === 🛍️ Top Categories ===
        if 'sub_category' in dataset.columns:
            st.write("#### Top Categories" if language == "English" else "#### Danh mục phổ biến")
            value = value = st.slider("Select the number of category to display" if language == "English" else "Chọn số lượng thể loại để hiển thị", min_value=0, max_value=dataset["sub_category"].nunique(), value=5)
            top_categories = dataset['sub_category'].value_counts().head(value)
            st.bar_chart(top_categories)
        
        st.markdown("---")

        # === ☁️ WordCloud for Product Names ===
        if language == "English":
            st.write("#### ☁️ Word Cloud of Product Names")
        else:
            st.write("#### ☁️ Word Cloud từ của Tên sản phẩm")

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join(dataset['product_name'].astype(str).tolist()))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        st.markdown("---")
        if 'price' in dataset.columns:
            if language == "English":
                st.write(f"#### 💹💹💹 Price Distribution {selected_dataset}")
            else:
                st.write(f"#### 💹💹💹 Phân phối giá cả cho {selected_dataset}")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "⭐ Max" if language == "English" else "⭐ Cao nhất", 
                f"{int(dataset['price'].max()):,}".replace(",", ".") + " đ"
            )

            col2.metric(
                "🔻 Min" if language == "English" else "🔻 Thấp nhất", 
                f"{int(dataset['price'].min()):,}".replace(",", ".") + " đ"
            )

            col3.metric(
                "📊 Mean" if language == "English" else "📊 Trung bình", 
                f"{int(dataset['price'].mean()):,}".replace(",", ".") + " đ"
            )

        
        fig2 = px.box(
            dataset,
            x='price',
            points='outliers',  # Only show outliers
            hover_data=["product_name", "rating", "price"],
            title=" ",
        )
        fig2.update_traces(
            marker=dict(
                color="#1f77b4",  # deep blue
                size=8,
                line=dict(width=1.5, color='#ffffff')  # white border
            )
        )

        fig2.update_layout(
            xaxis=dict(
                title= " ",
                title_font=dict(size=20, color="#1f77b4"),
                tickfont=dict(size=16, color="#1f77b4"),
                showgrid=True,
                gridcolor="rgba(173, 216, 230, 0.3)",  # light blue grid
                linecolor="#1f77b4",
                zerolinecolor="#1f77b4"
            ),
            yaxis=dict(
                title=None,
                tickfont=dict(size=16, color="#1f77b4"),
                showgrid=True,
                gridcolor="rgba(173, 216, 230, 0.3)",
                linecolor="#1f77b4",
                zerolinecolor="#1f77b4"
            ),
            plot_bgcolor='rgba(240, 248, 255, 0.6)',  # light blueish background
            paper_bgcolor='rgba(240, 248, 255, 1)',   # light paper color
            font=dict(family="Roboto, sans-serif", size=18, color="#1f77b4"),
            margin=dict(l=40, r=40, t=70, b=40),
            height=500
        )


        st.plotly_chart(fig2, use_container_width=True)


    else: 
    # === 👥 User Insights ===
        if language == "English":
            st.write("#### 👥 User Activity")
        else:
            st.write("#### 👥 Hoạt động người dùng")

        if 'user_id' in dataset.columns:
            user_info = dataset[['user_id', 'user']].drop_duplicates()
            user_counts = dataset['user_id'].value_counts().reset_index()
            user_counts.columns = ['user_id', 'activity_count']

            # Merge to get user names
            user_counts = pd.merge(user_counts, user_info, on='user_id', how='left')

            st.write("*Total Users*:" if language == "English" else "*Tổng số lượng người dùng*", user_counts.shape[0])

            # Selection box to choose what to view
            option = st.selectbox(
                "Choose user activity to display:" if language == "English" else "Chọn hoạt động đánh giá của người dùng",
                ("Most Active Users" if language == "English" else "Những người dùng đánh giá nhiều nhất", "Least Active Users" if language == "English" else "Những người dùng đánh giá ít nhất")
            )

            # Sort based on selection
            if option == "Most Active Users" or option == "Những người dùng đánh giá nhiều nhất": 
                display_data = user_counts.sort_values(by="activity_count", ascending=False).head(10)
            else:
                display_data = user_counts.sort_values(by="activity_count", ascending=True).head(10)

            st.subheader(option)
            st.dataframe(display_data[['user_id', 'user', 'activity_count']])

    st.markdown("---")
    # === ⭐ Rating Distribution (for fashion dataset) ===
    if 'rating' in dataset.columns:
        if language == "English":
            st.write(f"#### ⭐⭐⭐ Ratings Distribution {selected_dataset}")
        else:
            st.write(f"#### ⭐⭐⭐ Phân phối Rating cho {selected_dataset}")

        col1, col2, col3 = st.columns(3)

        col1.metric("⭐ Max" if language == "English" else "⭐ Cao nhất", f"{dataset['rating'].max():.2f}")
        col2.metric("🔻 Min" if language == "English" else "🔻 Thấp nhất", f"{dataset['rating'].min():.2f}")
        col3.metric("📊 Mean" if language == "English" else "📊 Trung bình", f"{dataset['rating'].mean():.2f}")

        fig, ax = plt.subplots()
        sns.countplot(data=dataset, x='rating', ax=ax, order=sorted(dataset['rating'].unique()))
        ax.set_title("Rating Distribution" if language == "English" else "Phân phối số sao")
        if (selected_dataset == "Products Ratings"):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
        st.pyplot(fig)

# else:
#     st.write("""---""")
#     mean_ratings_by_subcategory = dataset.groupby('sub_category')[['user_rating', 'product_overall_rating']].mean()
#     import plotly.graph_objects as go

#     fig1 = go.Figure()

#     # Line for user rating
#     fig1.add_trace(go.Scatter(
#         x=mean_ratings_by_subcategory.index,
#         y=mean_ratings_by_subcategory['user_rating'],
#         mode='lines+markers',
#         name='User Rating' if language == "English" else "Đánh giá người dùng",
#         line=dict(color='royalblue', width=3),
#         marker=dict(size=6)
#     ))

#     # Line for product overall rating
#     fig1.add_trace(go.Scatter(
#         x=mean_ratings_by_subcategory.index,
#         y=mean_ratings_by_subcategory['product_overall_rating'],
#         mode='lines+markers',
#         name='Product Overall Rating' if language == "English" else "Đánh giá chung của sản phẩm",
#         line=dict(color='orange', width=3),
#         marker=dict(size=6)
#     ))

#     fig1.update_layout(
#         title="📈 Mean Ratings by Sub-Category" if language == "English" else "📈 Đánh giá trung bình theo chủng loại",
#         xaxis_title="Sub-Category",
#         yaxis_title="Mean Rating",
#         xaxis=dict(tickangle=-60),
#         legend=dict(title="Legend", orientation="v", yanchor="bottom", y=1, xanchor="right", x=1),
#         plot_bgcolor='white',
#         margin=dict(l=40, r=40, t=60, b=80),
#         height=500
#     )


#     st.plotly_chart(fig1, use_container_width=True)

#     st.write("""---""")
#     # Calculate the count of 1-star and 5-star ratings
#     rating_counts = dataset.groupby('sub_category')['user_rating'].value_counts().unstack(fill_value=0)

#     # Initialize figure
#     fig = go.Figure()

#     # Add 1-star line
#     fig.add_trace(go.Scatter(
#         x=rating_counts.index,
#         y=rating_counts[1],
#         mode='lines+markers',
#         name='⭐ 1-Star Ratings' if language == "English" else "⭐ Đánh giá 1 sao",
#         line=dict(color='red', width=3),
#         marker=dict(size=6)
#     ))

#     # Add 5-star line
#     fig.add_trace(go.Scatter(
#         x=rating_counts.index,
#         y=rating_counts[5],
#         mode='lines+markers',
#         name='🌟 5-Star Ratings'if language == "English" else "⭐ Đánh giá 5 sao",
#         line=dict(color='green', width=3),
#         marker=dict(size=6)
#     ))

#     # Update layout for better aesthetics
#     fig.update_layout(
#         title="📊 1-Star and 5-Star Ratings by Sub-Category" if language == "English" else "📊 Thống kê đánh giá 1 sao và 5 sao theo thể loại sản phẩm",
#         xaxis_title="Sub-Category",
#         yaxis_title="Number of Ratings",
#         xaxis=dict(tickangle=-60),
#         legend=dict(title="Legend", orientation="v", yanchor="bottom", y=1, xanchor="right", x=1),
#         plot_bgcolor='white',
#         height=500,
#         margin=dict(l=40, r=40, t=60, b=100)
#     )

#     # Display in Streamlit
#     st.plotly_chart(fig, use_container_width=True)

# === 📥 Download Option ===
if language == "English":
    st.sidebar.download_button(f"📥 Download dataset {selected_dataset}", dataset.to_csv(index=False), dataset_name_products if selected_dataset == "Products Ratings" else dataset_name_user_rating)
else:
    st.sidebar.download_button(f"📥 Tải xuống dữ liệu {selected_dataset}", products_df.to_csv(index=False), dataset_name_products if selected_dataset == "Products Ratings" else dataset_name_user_rating)
