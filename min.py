import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

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
            st.experimental_rerun() #to change state in only one click

    with col2:
        st.markdown("<p class='medium-font'>Ready for a deeper dive? Unlock more features like multiple hidden layers and dropout options to enhance your neural network understanding.</p>", unsafe_allow_html=True)
        if st.button('Advanced'):
            st.session_state['page'] = 'Advanced'
            st.experimental_rerun() #to change state in only one click


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

def build_advanced_model(activation, neurons_per_layer, dropout_rate):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # Flatten the input

    # Add layers based on the dictionary
    for layer, neurons in neurons_per_layer.items():
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation='softmax'))  # Output layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_advanced_cnn_model(activation, neurons_per_layer, filters, kernel_size, dropout_rate):
    model = Sequential()
    
    # Convolutional Layer
    model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), activation=activation, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # Dense Layers based on the dictionary
    model.add(Flatten())
    for layer, neurons in neurons_per_layer.items():
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(10, activation='softmax'))  # Output layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def advanced_page():
    if st.button('Go Back to Home'):
        st.session_state['page'] = 'Home'
        st.experimental_rerun()

    st.title('Advanced Neural Network Exploration')

    model_type = st.selectbox('Choose model type:', ['Standard Neural Network', 'CNN'])

    activation = st.selectbox('Choose activation function:', ['relu', 'sigmoid', 'tanh'])
    layers = st.slider('Select number of layers:', 1, 5, 2)

    neurons_per_layer = {}
    for i in range(layers):
        neurons_per_layer[f'layer_{i+1}'] = st.slider(f'Select number of neurons for Layer {i+1}:', 32, 512, 128, 32)

    dropout_rate = st.slider('Select dropout rate:', 0.0, 0.9, 0.5, 0.1)

    if model_type == 'CNN':
        filters = st.slider('Select number of filters:', 16, 128, 32, 16)
        kernel_size = st.slider('Select kernel size:', 2, 5, 3)

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

    if st.button('Start Training'):
        with st.spinner('Training in progress...'):
            if model_type == 'Standard Neural Network':
                model = build_advanced_model(activation, neurons_per_layer, dropout_rate)
            else:
                model = build_advanced_cnn_model(activation, neurons_per_layer, filters, kernel_size, dropout_rate)
            history = model.fit(st.session_state['train_images'], st.session_state['train_labels'],
                                validation_split=test_size, epochs=epochs, verbose=0)
            st.session_state['model'] = model
            st.session_state['training_history'] = history
            st.session_state['trained'] = True
            st.success('Training completed!')


    # [Rest of the visualization and prediction code]

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


def visualize_data(images, labels):
    st.write("Sample Images from the Dataset:")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {np.argmax(labels[i])}')
        ax.axis('off')
    st.pyplot(fig)

def plot_training_history(history):
    st.write("Training and Validation Metrics:")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, acc, label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()

    ax2.plot(epochs_range, loss, label='Training Loss')
    ax2.plot(epochs_range, val_loss, label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()

    st.pyplot(fig)

if __name__ == '__main__':
    main()