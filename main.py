import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

# Load .env variables if any
load_dotenv()

st.title("Insight Flow: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
