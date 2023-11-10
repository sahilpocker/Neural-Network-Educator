from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import streamlit as st

@st.cache_data
def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels
