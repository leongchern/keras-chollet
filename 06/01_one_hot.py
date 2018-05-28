import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
            max_length = 10
            results = np.zeros(shape=(len(samples),
                                      max_length,
                                      max(token_index.values()) + 1))

print(token_index)
print(token_index.values())
print(results)

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
    # for j, word in list(enumerate(sample.split())):
        index = token_index.get(word)
        results[i, j, index] = 1.

# print(token_index)
# print(token_index.values())
# print(results)