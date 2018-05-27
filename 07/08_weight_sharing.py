"""
    Layer weight sharing in Keras
"""

from tensorflow.python.keras import layers, Input
from tensorflow.python.keras.models import Model
import numpy as np


# Generate dummy data
left_data = np.random.random((1000, 128))
left_data = left_data.reshape(1000, -1, 128)
right_data = np.random.random((1000, 128))
right_data = right_data.reshape(1000, -1, 128)
targets = np.random.random(1000)
targets = targets.reshape(1000, 1)


# Network design
lstm = layers.LSTM(32)

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)    # Re-using the left branch's LSTM weights

merged = layers.concatenate([left_output, right_output], axis = -1)
preds = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], preds)

model.compile(optimizer='rmsprop', loss='mse')
model.fit([left_data, right_data], targets, epochs=3, batch_size=8)

