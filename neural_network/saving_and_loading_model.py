import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# ignore some output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # GPU device
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Make sure we don't get any GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

# 1. How to save and load model weights
#   - Need to load it exactly the way saved it
#
# 2. Save and load entire model (Serializing model)
#   - Save weights
#   - Model architecture
#   - Training configuration (model.compile())
#   - Optimizer and states

model1 = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

inputs = keras.Input(shape=28*28)
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
model2 = keras.Model(inputs=inputs, outputs=outputs)


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10)

    def call(self, input_tensor, training=None, mask=None):
        input_x = tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(input_x)


model3 = MyModel()

model = model2
# load saved model weights
# model.load_weights('saved_model/')
# load entire model
# model = keras.models.load_model('saved_entire_model/')

model.summary()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(X_test, y_test, batch_size=32, verbose=2)

# save model weights
model.save_weights('saved_model/')
# save entire model
model.save('saved_entire_model/')
