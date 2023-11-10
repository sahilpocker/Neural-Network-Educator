import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LambdaCallback
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Define the preprocess function
def preprocess_image(image, target_size=(28, 28)):
    # Convert to grayscale and invert the colors
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Model expects a batch of images to process
    return image_array

# Define the model architecture with a parameter for activation and number of neurons
def build_model(activation='relu', neurons=128):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(neurons, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load and cache the MNIST data
@st.cache_data
def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

# Streamlit UI
def main():
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

    # Start training button
    if st.button('Start Training'):
        if not st.session_state.get('data_loaded', False):
            st.error('Error: Data not loaded. Please load the data before training.')
        else:
            with st.spinner('Training in progress...'):
                model = build_model(activation, neurons)
                model.fit(st.session_state['train_images'], st.session_state['train_labels'],
                          validation_split=test_size, epochs=epochs, verbose=0)
                st.session_state['model'] = model
                st.success('Training completed!')


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

if __name__ == '__main__':
    main()