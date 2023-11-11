
---

# Neural Network Educator

## Introduction
Neural Network Educator is an interactive web application designed to teach the fundamentals of neural networks. It uses the MNIST dataset to demonstrate handwritten digit classification and provides both beginner and advanced modes for users to explore and learn.

## Live Application
Experience the Neural Network Educator live [here](https://neuralneteducator.streamlit.app/).

## Features
- **Beginner Mode:** Start with basic concepts, explore datasets, and build simple models.
- **Advanced Mode:** Dive deeper with features like multiple hidden layers, dropout options, and convolutional neural networks (CNNs).
- **Interactive UI:** Draw digits and see real-time predictions from the trained model.
- **Data Visualization:** View sample images from the dataset and training/validation metrics.


---

## Modules

The Neural Network Educator application is organized into several modules, each serving a specific purpose in the overall functionality of the app. Below is the tree structure of the project, detailing the organization and responsibilities of each module.

```
neural_network_app/
│
├── min.py                   # Main Application Entry Point
├── image_preprocessing.py   # Image Preprocessing Functions
├── model.py                 # Neural Network Model Building Functions
├── data_loader.py           # MNIST Dataset Loading and Preprocessing
├── home_page.py             # Home Page UI Components
├── beginner_page.py         # Beginner Page UI Components
├── advanced_page.py         # Advanced Page UI Components
├── data_visualization.py    # Data Visualization Functions
└── training_history.py      # Training History Visualization Functions
```

### Module Descriptions

- **app.py**: This is the main entry point of the application. It uses Streamlit to render the UI and manages navigation between different pages of the app.

- **image_preprocessing.py**: Contains functions for preprocessing images, such as converting canvas drawings to a format suitable for model predictions.

- **model.py**: Includes functions to build different types of neural network models, including basic and advanced configurations.

- **data_loader.py**: Responsible for loading and preprocessing the MNIST dataset, utilizing caching for efficiency.

- **home_page.py**: Defines the UI components and logic for the home page of the application, presenting the initial interface to the user.

- **beginner_page.py**: Contains the UI elements and interactivity for the beginner's section, allowing users to experiment with basic neural network concepts.

- **advanced_page.py**: Manages the UI for the advanced section, providing more complex functionalities and model configurations for experienced users.

- **data_visualization.py**: Includes functions for visualizing data, such as displaying sample images from the MNIST dataset.

- **training_history.py**: Provides functionalities to plot and visualize the training history of the neural network models.

---

## Installation and Usage
To run the application locally, follow these steps:
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.

## Contributing
Contributions to the Neural Network Educator are welcome. Please read the contributing guidelines before submitting your changes.

## License
This project is licensed under the [MIT License](LICENSE).

---