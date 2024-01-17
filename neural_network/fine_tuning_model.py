import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

# ignore some output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # GPU device
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# HYPERPARAMETERS
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3

# Make sure we don't get any GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# load the pretrained model
model = keras.models.load_model('saved_entire_pretrained_model/')
# freeze all layers
model.trainable = False
# freeze individual layer
for layer in model.layers:
    assert layer.trainable is False
    layer.trainable = False

base_inputs = model.layers[0].input
base_outputs = model.layers[-1].output
# customize the final output layer
final_outputs = layers.Dense(10)(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)

# ======================================================== #
#                   Pretrained Keras Model                 #
# ======================================================== #
X = tf.random.normal(shape=(5, 299, 299, 3))
y = tf.constant([0, 1, 2, 3, 4])

model = keras.applications.InceptionV3(include_top=True)
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(5)(base_outputs)
new_model1 = keras.Model(inputs=base_inputs, outputs=final_outputs)






model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.legacy.Adam(),
    metrics=['accuracy'],
)

model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)

# save model weights
model.save_weights('saved_model/')
# save entire model
model.save('saved_entire_model/')
