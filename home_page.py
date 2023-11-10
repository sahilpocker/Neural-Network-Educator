import streamlit as st

def home_page():
    st.write("""
    <style>
    .big-font {
        font-size:30px !important;
        font-family: 'Helvetica', sans-serif;
    }
    .medium-font {
        font-size:20px !important;
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Welcome to our Neural Network Education App!</p>', unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Whether you're just starting or have some experience, we've tailored our learning paths for your needs. Choose your expertise level below to begin exploring the fascinating world of neural networks.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    
    with col1:
        st.markdown("<p class='medium-font'>New to neural networks? Start here to explore datasets, build basic models, and test your understanding with simplified parameters.</p>", unsafe_allow_html=True)
        if st.button('Beginner'):
            st.session_state['page'] = 'Beginner'
            st.experimental_rerun()

    with col2:
        st.markdown("<p class='medium-font'>Ready for a deeper dive? Unlock more features like multiple hidden layers and dropout options to enhance your neural network understanding.</p>", unsafe_allow_html=True)
        if st.button('Advanced'):
            st.session_state['page'] = 'Advanced'
            st.experimental_rerun()
