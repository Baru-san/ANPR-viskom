import streamlit as st

st.set_page_config(
    page_title="Image Editing",
    
)

st.write("# Image Editing")

st.sidebar.success("SIlahkan Pilih Metode Pengeditan")

st.markdown(
    """
    **Histogram Equalization**
    **Face Blurring**
    **Edge Detection**
"""
)