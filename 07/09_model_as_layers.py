"""
    Model as layers using Keras
"""

import os
import numpy as np
from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras import layers, Input
from tensorflow.python.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Dummy data
left_data = np.random.random((100,25, 25, 3))
right_data = np.random.random((100,25, 25, 3))
targets = np.random.random((100, 1, 1, 1))

# Model
xception_base = Xception(weights=None, include_top=False)

left_input = Input(shape=(25, 25, 3))
right_input = Input(shape=(25, 25, 3))

left_features = xception_base(left_input)
right_features = xception_base(right_input)

merged_features = layers.concatenate([left_features, right_features], axis=-1)
preds = layers.Dense(1, activation='sigmoid')(merged_features)

model = Model([left_input, right_input], preds)

model.compile(optimizer='rmsprop', loss='mse')
model.fit([left_data, right_data], targets, batch_size=8, epochs=3)




