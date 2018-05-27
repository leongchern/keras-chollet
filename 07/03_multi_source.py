"""
    Functional API - multiple inputs, single output
"""

from tensorflow.python.keras import Input, layers
from tensorflow.python.keras.models import Model
import numpy as np

text_vocab_size = 10000
question_vocab_size = 10000
answer_vocab_size = 500

# Generating dummy data for demo
num_samples = 1000
max_len = 100
text = np.random.randint(1, text_vocab_size, size=(num_samples, max_len))
question = np.random.randint(1, question_vocab_size, size=(num_samples, max_len))
answers = np.random.randint(0, 2, size=(num_samples, answer_vocab_size))


text_input = Input(shape=(None, ), dtype='int32', name='text')
# embedded_text = layers.Embedding(64, text_vocab_size)(text_input)
embedded_text = layers.Embedding(text_vocab_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None, ), dtype='int32', name='question')
# embedded_question = layers.Embedding(32, question_vocab_size)(question_input)
embedded_question = layers.Embedding(question_vocab_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
# print(concatenated.shape)       # 32 + 16 which gives us (?, 48) ... good
answer = layers.Dense(answer_vocab_size, activation='softmax')(concatenated)


model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit([text, question], answers, epochs=10, batch_size=128)
