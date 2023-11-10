import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def visualize_data(images, labels):
    st.write("Sample Images from the Dataset:")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {np.argmax(labels[i])}')
        ax.axis('off')
    st.pyplot(fig)
