from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers


max_features = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


mdl = Sequential()
mdl.add(layers.Embedding(10000, 8, input_length=max_len))
mdl.add(layers.Flatten())
mdl.add(layers.Dense(1, activation='sigmoid'))
mdl.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])
mdl.summary()

hist = mdl.fit(x_train, y_train,
               epochs=10,
               batch_size=32,
               validation_split=.2)

