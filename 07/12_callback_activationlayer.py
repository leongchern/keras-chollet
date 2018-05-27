"""
    Writing own callbacks
"""

import os
import numpy as np
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.callbacks import Callback


class ActivationLogger(Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Generate fake data
x_train = np.random.random((1000,2))
y_train = np.random.random((1000))
x_val = np.random.random((100,2))
y_val = np.random.random((100))


# Model
callbacks_list = [
    ActivationLogger
]


mdl = models.Sequential()
mdl.add(layers.Dense(16, activation='relu'))
mdl.add(layers.Dense(8, activation='relu'))
mdl.add(layers.Dense(1, activation='sigmoid'))

mdl.compile(optimizer='rmsprop', loss='binary_crossentropy')
mdl.fit(x_train, y_train,
        batch_size=8,
        epochs=10,
        callbacks=callbacks_list,
        validation_data=[x_val, y_val])

