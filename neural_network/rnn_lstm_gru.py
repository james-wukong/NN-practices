import os
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# ignore some output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # GPU device
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = keras.Sequential()
# shape: n_steps, n_features
model.add(keras.Input(shape=(None, 28)))
model.add(
    # multiple hidden LSTM layers can be stacked one on top of another
    # in what is referred to as a stacked LSTM model.
    # an LSTM layer requires a THREE-Dim input and LSTMs by default
    # will produce TWO-Dim output as an interpretation from the end of the sequence.
    # we can address this by having the LSTM output a value for each time step
    # in the input data by setting the 'return_sequence=True' argument on the layer.
    # This allows us to have 3D output from hidden LSTM layer as input to the next.
    layers.SimpleRNN(512, return_sequences=True, activation='relu')
    # layers.GRU(256, return_sequences=True, activation='tanh')
    # layers.LSTM(256, return_sequences=True, activation='tanh')
    # layers.Bidirectional(
    #     layers.LSTM(512, return_sequences=True, activation='tank')
    # )
)
model.add(
    layers.SimpleRNN(512, activation='relu')
    # layers.GRU(256, activation='tanh')
    # layers.LSTM(256, activation='tanh')
    # layers.Bidirectional(
    #     layers.LSTM(512, activation='tank')
    # )
)
model.add(layers.Dense(10))

model.summary()

model.compile(
    # add 'from_logits=True' when not using softmax activation function
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
    metrics=['Accuracy'],
)
model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)
