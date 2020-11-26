import streamlit as st

st.title("my first app")
text = st.text_input('Enter text here...')

st.write(text)