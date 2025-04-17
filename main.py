import streamlit as st
from streamlit_extras.switch_page_button import switch_page  # optional for page control
from PIL import Image

# Set language (stores in session state)
if "language" not in st.session_state:
    st.session_state.language = "Tiếng Việt"


image = Image.open('shopee.jpg')
st.sidebar.image(image, caption='Shopee Recommendation System', use_container_width=True)
# st.session_state.language = st.sidebar.radio("🌐 Select Language / Chọn ngôn ngữ", ("English", "Tiếng Việt"))
# Language selector with key, non-destructive
selected_lang = st.sidebar.radio(
    "🌐 Select Language / Chọn ngôn ngữ", 
    ("English", "Tiếng Việt"), 
    index=("English", "Tiếng Việt").index(st.session_state.language),
    key="language_selector"
)

# Only update session state if language changed
if selected_lang != st.session_state.language:
    st.session_state.language = selected_lang
    st.rerun()  # Optional: force refresh so UI reflects new language immediately
st.sidebar.write("""---""")

# Define your pages (as you did)
introduction = st.Page("introduction.py", title="Introduction", icon="🎈")
user_guide = st.Page("user_guide.py", title="User Guide", icon="❄️")
eda = st.Page("eda.py", title="Exploratory Data Analysis", icon="🎉")
recommendation = st.Page("recommendation.py", title="Recommendation System", icon="🤖")

pg = st.navigation([introduction, user_guide, eda, recommendation])
pg.run()
st.sidebar.write("""Made By 
                 
                    Bùi Khánh An & Trần Thanh Trúc""")
st.sidebar.write("""Instructed By 
                 
                    ♥️♥️♥️ Khuất Thuỳ Phương ♥️♥️♥️""")

st.sidebar.markdown("# April 2025")
