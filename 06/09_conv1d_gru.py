from tensorflow.python.keras import layers
from tensorflow.python.keras import models

mdl = models.Sequential()
mdl.add(layers.Conv1D(32, 5, activation='relu',
               input_shape=(20, 10)))
mdl.add(layers.MaxPooling1D(2))
mdl.add(layers.Conv1D(32, 5, activation='relu'))

mdl.summary()
