from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt


max_features = 10000
max_len = 500
batch_size = 128
num_epochs = 10


(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time')
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# Training model
mdl = Sequential()
mdl.add(Embedding(max_features, 32))
mdl.add(SimpleRNN(32))
mdl.add(Dense(1, activation='sigmoid'))

mdl.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])
hist = mdl.fit(input_train, y_train,
               epochs=num_epochs,
               batch_size=batch_size,
               validation_split=0.2)


# Plot results
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
