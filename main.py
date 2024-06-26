from lm import generator
import streamlit as st

st.title("Random Name Generator")

if st.button("Generate"):
    name=generator()
    st.text_area("Name Generated",name)