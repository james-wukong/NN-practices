import os
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# ignore some output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

model = keras.Sequential(
    [
        keras.Input(shape=(28*28,), name='input_layer'),
        layers.Dense(units=512, activation='relu', name='first_hidden_layer'),
        layers.Dense(units=256, activation='relu', name='second_hidden_layer'),
        layers.Dense(units=10, activation='softmax')
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=1, verbose=2)
