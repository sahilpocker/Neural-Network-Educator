from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

def build_model(activation='relu', neurons=128):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(neurons, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_advanced_model(activation, neurons_per_layer, dropout_rate):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for layer, neurons in neurons_per_layer.items():
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_advanced_cnn_model(activation, neurons_per_layer, filters, kernel_size, dropout_rate):
    model = Sequential()
    model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), activation=activation, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for layer, neurons in neurons_per_layer.items():
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
