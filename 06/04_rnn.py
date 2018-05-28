from tensorflow.python.keras.layers import SimpleRNN, Embedding
from tensorflow.python.keras.models import Sequential


# Return output at last cell
print('\n\n\nOutput at last rnn cell only')
mdl = Sequential()
mdl.add(Embedding(10000, 32))
mdl.add(SimpleRNN(32))
mdl.summary()


# Return all values at intermediate cells
print('\n\n\nOutput for each and every rnn cells')
mdl2 = Sequential()
mdl2.add(Embedding(10000, 32))
mdl2.add(SimpleRNN(32, return_sequences=True))
mdl2.summary()


# Stacked RNN
print('\n\n\nOutput at each and every rnn cells (stacked network)')
mdl3 = Sequential()
mdl3.add(Embedding(10000, 32))
mdl3.add(SimpleRNN(32, return_sequences=True))
mdl3.add(SimpleRNN(32, return_sequences=True))
mdl3.add(SimpleRNN(32, return_sequences=True))
mdl3.add(SimpleRNN(32))
mdl3.summary()
