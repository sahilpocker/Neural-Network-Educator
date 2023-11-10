import streamlit as st
from data_loader import load_mnist_data
from model import build_model
from data_visualization import visualize_data
from training_history import plot_training_history
from image_processing import preprocess_image
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image


def beginner_page():

    if st.button('Go Back to Home'):
        st.session_state['page'] = 'Home'
        st.experimental_rerun()


    st.title('Handwritten Digit Classification with MNIST')

    # Choice of activation function
    activation = st.selectbox('Choose activation function:', ['relu', 'sigmoid', 'tanh'])

    # Slider for number of neurons in the hidden layer
    neurons = st.slider('Select number of neurons in the hidden layer:', min_value=32, max_value=512, value=128, step=32)

    # Slider for test-train split
    test_size = st.slider('Select Test Size Ratio:', min_value=0.1, max_value=0.9, value=0.2, step=0.05)

    # Slider for number of epochs
    epochs = st.slider('Select Number of Epochs:', min_value=1, max_value=100, value=5)

    # Button to load data
    if st.button('Load Data'):
        st.session_state['train_images'], st.session_state['train_labels'], \
        st.session_state['test_images'], st.session_state['test_labels'] = load_mnist_data()
        st.session_state['data_loaded'] = True
        st.success('Data loaded successfully!')

    if st.button('Visualize Data'):
        st.session_state['visualize'] = True
        st.session_state['trained'] = False

    if 'visualize' in st.session_state and st.session_state['visualize']:
        if 'train_images' in st.session_state and 'train_labels' in st.session_state:
            visualize_data(st.session_state['train_images'], st.session_state['train_labels'])
        else:
            st.error('Data not loaded. Please load the data first.')


    # Start training button
    if st.button('Start Training'):
            st.session_state['visualize'] = False  # Hide visualization
            with st.spinner('Training in progress...'):
                model = build_model(activation, neurons)
                history = model.fit(st.session_state['train_images'], st.session_state['train_labels'],
                                    validation_split=test_size, epochs=epochs, verbose=0)
                st.session_state['model'] = model
                st.session_state['training_history'] = history
                st.session_state['trained'] = True
                st.success('Training completed!')

    if 'trained' in st.session_state and st.session_state['trained']:
        plot_training_history(st.session_state['training_history'])


    # Canvas for drawing the digit
    st.write('Draw a digit below and press Submit:')
    canvas_result = st_canvas(
        fill_color='white',
        stroke_width=10,
        stroke_color='black',
        background_color='white',
        height=150,
        width=150,
        drawing_mode='freedraw',
        key='canvas'
    )

    # Submit button for prediction
    if st.button('Submit'):
        if canvas_result.image_data is not None:
            if 'model' in st.session_state:
                # Convert the canvas result into a PIL Image and predict
                image = Image.fromarray((canvas_result.image_data).astype('uint8'), mode='RGBA')
                preprocessed_image = preprocess_image(image)
                prediction = st.session_state['model'].predict(preprocessed_image)
                st.write('Predicted digit:', np.argmax(prediction))
            else:
                st.error('Please train the model before making predictions.')
        else:
            st.error('Please draw a digit to predict.')