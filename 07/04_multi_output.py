"""
    Functional API - single input, multiple outputs
"""

from tensorflow.python.keras import layers, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
import numpy as np

vocab_size = 50000
num_income_groups = 10

# Dummy data for demo
posts = np.random.randint(10000, size=(70000, 5))
age_targets = np.random.randint(50, size=(70000, 1))
income_targets = np.random.randint(0, num_income_groups, size=posts.shape[0])
income_targets = to_categorical(income_targets)
gender_targets = np.random.randint(0, 2, size=posts.shape[0])
gender_targets = to_categorical(gender_targets)

# Functional model
posts_inputs = Input(shape=(5, ), dtype='float32', name='posts')
# embedded_posts = layers.Embedding(256, vocab_size)(posts_inputs)
# x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.MaxPooling1D(5)(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.Conv1D(256, 5, activation='relu')(x)
# x = layers.GlobalMaxPooling1D()(x)
# x = layers.Dense(128, activation='relu')(x)

embedded_posts = layers.Dense(64, activation='sigmoid')(posts_inputs)
x = layers.Dense(64, activation='sigmoid')(embedded_posts)
x = layers.Dense(16, activation='sigmoid')(x)


# Preds
age_pred = layers.Dense(1, name='age')(x)
income_pred = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_pred = layers.Dense(2, activation='softmax', name='gender')(x)

model = Model(posts_inputs, [age_pred, income_pred, gender_pred])

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)




