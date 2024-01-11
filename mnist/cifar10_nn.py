import os
# import tensorflow as tf
# # GPU device
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

# ignore some output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = keras.Sequential(
    [
        # Defines the model's input layer, expecting images of size 32x32 pixels with 3 color channels (RGB).
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
        # Performs max pooling with a pool size of 2x2, reducing spatial dimensions.
        layers.MaxPool2D(pool_size=(2, 2)),
        # Applies 64 filters of size 3x3 with ReLU activation.
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),
        # Applies 128 filters of size 3x3 with ReLU activation.
        layers.Conv2D(128, 3, activation='relu'),
        # Flattens the 3D output of convolutional layers into a 1D vector for dense layers.
        layers.Flatten(),
        # Fully connected layer with 64 neurons and ReLU activation.
        layers.Dense(64, activation='relu'),
        # Final output layer with 10 neurons (likely for 10-class classification).
        layers.Dense(10),
    ]
)

model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
    metrics=['accuracy'],
)

model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)


def my_model() -> keras.Model:
    # Creates an input layer expecting images with dimensions 32x32 pixels and 3 color channels (RGB).
    inputs = keras.Input(shape=(32, 32, 3))
    # Applies 32 filters of size 3x3 to extract features from the input images.
    x = layers.Conv2D(32, 3, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(inputs)
    # Normalizes the activations of the previous layer to improve training stability and speed.
    x = layers.BatchNormalization()(x)
    # Applies the ReLU activation function to introduce non-linearity.
    x = keras.activations.relu(x)
    # Performs max pooling with a pool size of 2x2 to reduce spatial dimensions and capture essential features.
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 5, padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)

    return keras.Model(inputs=inputs, outputs=outputs)


model = my_model()
model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=3e-4),
    metrics=['accuracy'],
)

model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
