import streamlit as st
from home_page import home_page
from beginner_page import beginner_page
from advanced_page import advanced_page

def main():
    st.set_page_config(layout="wide")

    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'

    page = st.session_state['page']

    if page == "Home":
        home_page()
    elif page == "Beginner":
        beginner_page()
    elif page == "Advanced":
        advanced_page()

if __name__ == '__main__':
    main()
