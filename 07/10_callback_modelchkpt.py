"""
    Experimenting with modelcheckpoint and earlystopping callbacks
"""

import os
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


os.environ['TF_CP_MIN_LOG_LEVEL']='2'


callbacks_list = [
    EarlyStopping(
        monitor='acc',
        patience=1
    ),
    ModelCheckpoint(
        filepath='files/chkpt.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]


# Generate fake data
x = np.random.random((1000,2))
y = np.random.random((1000))
val_x = np.random.random((100,2))
val_y = np.random.random((100))


# Model
model = Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x, y,
          epochs=10,
          batch_size=8,
          callbacks=callbacks_list,
          validation_data=[val_x, val_y])