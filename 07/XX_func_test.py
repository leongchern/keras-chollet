"""
    Functional API - single input, multiple outputs
"""

import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers, Input

data = np.random.random(size=(1000,10))
y_targets = np.random.random(1000)

input_data = Input(shape=(10, ), dtype='float32')
x = layers.Dense(16, activation='sigmoid')(input_data)
x = layers.Dense(8, activation='sigmoid')(x)
y_pred = layers.Dense(1)(x)

model = Model(input_data, y_pred)

model.compile(optimizer='rmsprop', loss='mse')
model.fit(data, y_targets, epochs=5, batch_size=8)
