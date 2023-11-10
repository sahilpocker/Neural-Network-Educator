import streamlit as st

def home_page():
    # Inject custom CSS for overall page styling
    st.markdown("""
        <style>
            html, body, [class*="sidebar-"] {
                background-color: #E6E6FA; /* Background color */
            }
            .big-font {
                font-size:30px !important;
                font-family: 'Helvetica', sans-serif;
                color: black; /* Text color */
                text-align: center;
            }
            .medium-font {
                font-size:20px !important;
                font-family: 'Helvetica', sans-serif;
                color: black; /* Text color */
                text-align: center;
            }
            .st-bb {
                background-color: #C8A2C8; /* Secondary background color */
            }
            .st-at {
                background-color: #C8A2C8; /* Secondary background color */
            }
            .centered {
                display: flex;
                justify-content: center;
            }
        </style>
        """, unsafe_allow_html=True)

    # Display custom styled text
    st.markdown('<p class="big-font">Welcome to our Neural Network Education App!</p>', unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Whether you're just starting or have some experience, we've tailored our learning paths for your needs. Choose your expertise level below to begin exploring the fascinating world of neural networks.</p>", unsafe_allow_html=True)

    # Centered buttons within columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        if st.button('Beginner'):
            st.session_state['page'] = 'Beginner'
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        if st.button('Advanced'):
            st.session_state['page'] = 'Advanced'
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
