import streamlit as st
from streamlit_extras.switch_page_button import switch_page  # optional for page control

# Set language (stores in session state)
if "language" not in st.session_state:
    st.session_state.language = "English"

st.session_state.language = st.sidebar.radio("🌐 Select Language / Chọn ngôn ngữ", ("English", "Tiếng Việt"))


# Define your pages (as you did)
introduction = st.Page("introduction.py", title="Introduction", icon="🎈")
user_guide = st.Page("user_guide.py", title="User Guide", icon="❄️")
eda = st.Page("eda.py", title="Exploratory Data Analysis", icon="🎉")
recommendation = st.Page("recommendation.py", title="Recommendation System", icon="🤖")

pg = st.navigation([introduction, user_guide, eda, recommendation])
pg.run()
