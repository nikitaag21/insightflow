import os
import streamlit as st
import pickle
import time


st.title("Insight Flow: AI-Powered News Research for Equity Analysts 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
