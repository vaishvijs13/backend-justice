import streamlit as st

uploaded_video = st.file_uploader("Upload Deposition Video:", type=['mp4', 'wav'])

# for testing before connecting with frontend next