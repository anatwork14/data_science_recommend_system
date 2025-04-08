import streamlit as st
import numpy as np
import pandas as pd
import time


introduction = st.Page("introduction.py", title="Introduction", icon="🎈")
user_guide = st.Page("user_guide.py", title="User Guide", icon="❄️")
eda = st.Page("eda.py", title="Exploratory Data Analysis", icon="🎉")
recommendation = st.Page("recommendation.py", title="Recommendation System", icon="🤖")
# # Set up navigation
pg = st.navigation([introduction, user_guide, eda, recommendation])

pg.run()
