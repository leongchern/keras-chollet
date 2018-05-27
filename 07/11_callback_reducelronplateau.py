"""
    Experimenting with reducing LR callback
"""

import os
import numpy as np
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Generate fake data
x_train = np.random.random((1000,2))
y_train = np.random.random((1000))
x_val = np.random.random((100,2))
y_val = np.random.random((100))


# Model

callbacks_list = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10
    ),
    ModelCheckpoint(
        filepath='files/reduce_lr.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

mdl = models.Sequential()
mdl.add(layers.Dense(32, activation='relu'))
mdl.add(layers.Dense(16, activation='relu'))
mdl.add(layers.Dense(1, activation='sigmoid'))

mdl.compile(optimizer='rmsprop', loss='binary_crossentropy')
mdl.fit(x_train, y_train,
        batch_size=8,
        epochs=10,
        callbacks=callbacks_list,
        validation_data=[x_val, y_val])

