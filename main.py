import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import re
from tensorflow.python.keras.layers import LSTM, Embedding, Dropout, SpatialDropout1D, CuDNNLSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.constraints import maxnorm

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

"""
import data
"""
in_filename = "train.ft.txt"
in_filename_test = "test.ft.txt"
# in_filename = "head.txt"

tr = open(in_filename)
te = open(in_filename_test)
train_file_lines = tr.readlines()
test_file_lines = te.readlines()
tr.close()
te.close()

labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
texts = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

# convert digits to 0
for i in range(int(len(texts) / 4)):
    texts[i] = re.sub('\d', '0', texts[i])

# convert urls to "<url>"
for i in range(int(len(texts) / 4)):
    if 'www.' in texts[i] or 'http:' in texts[i] or 'https:' in texts[i] or '.com' in \
            texts[i]:
        texts[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", texts[i])

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
test_texts = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

# convert digits to 0
for i in range(int(len(test_texts)/4)):
    test_texts[i] = re.sub('\d', '0', test_texts[i])

# convert urls to "<url>"
for i in range(int(len(test_texts)/4)):
    if 'www.' in test_texts[i] or 'http:' in test_texts[i] or 'https:' in test_texts[i] or '.com' in \
            test_texts[i]:
        test_texts[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_texts[i])

max_len = 100
training_samples = 20000
validation_sample = 5000
# max words
max_features = 10000
batch_size = 2098

# labels = []
# texts = []
# labels_test = []
# texts_test = []
#
#
# def load_doc(filename, type):
#     ctn = 0
#     with open(filename, 'r') as reader:
#         line = reader.readline()
#         while line != '':
#             if line[:10] == '__label__1':
#                 if type == 'train':
#                     labels.append(0)
#                 else:
#                     labels_test.append(0)
#             else:
#                 if type == 'train':
#                     labels.append(1)
#                 else:
#                     labels_test.append(1)
#             line = reader.readline()
#             if type == 'train':
#                 texts.append(line[11:].strip())
#             else:
#                 texts_test.append(line[11:].strip())
#             ctn += 1
#
#
# load_doc(in_filename, 'train')
# load_doc(in_filename_test, 'test')

print('loading data...')

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.index_word
data = pad_sequences(sequences, maxlen=max_len)
label = np.asarray(labels)
print('Found %s unique tokens' % len(word_index))

# glove_dir = './glove.6B.100d.txt'
# def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
# embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(glove_dir))
#
# all_embs = np.stack(embeddings_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
# embed_size = all_embs.shape[1]
#
# word_index = tokenizer.word_index
# nb_words = min(max_features, len(word_index))
# #change below line if computing normal stats is too slow
# embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) #embedding_matrix = np.zeros((nb_words, embed_size))
# for word, i in word_index.items():
#     if i >= max_features: continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector

glove_dir = './'
embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embedding_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_features, embedding_dim))
for i, word in word_index.items():
    if i < max_features:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

"""
preprocess data
"""

"""
define model
"""

model = Sequential()
# model.add(Embedding(len(word_index) + 1, 32, input_length=200))
model.add(Embedding(max_features, embedding_dim, input_length=max_len, trainable=True))
# model.add(layers.Flatten())
# model.add(tf.keras.layers.LSTM(32, input_shape=(200, 32)))
# model.add(SpatialDropout1D(0.25))
# model.add(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
model.add(LSTM(32))
# model.add(LSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10),
#                     bias_constraint=maxnorm(10)))
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())

"""
train model
"""
opt = RMSprop(lr=0.001)
optTwo = Adam(lr=0.001)


def model_one():
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model.fit(data, label, epochs=10, batch_size=batch_size, validation_split=0.2, shuffle=True)


def model_two():
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model.fit(data, label, epochs=10, batch_size=batch_size, validation_split=0.2, shuffle=True,
                     validation_steps=4)


def model_three():
    model.compile(optimizer=optTwo, loss='binary_crossentropy', metrics=['acc'])
    return model.fit(data, label, epochs=10, batch_size=batch_size, validation_split=0.2, shuffle=True,
                     validation_steps=4)


def model_four():
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    return model.fit(data, label, epochs=10, batch_size=batch_size, validation_split=0.2, shuffle=True,
                     validation_steps=4)


model_1 = model_one()
# model_2 = model_two()
# model_3 = model_three()
# model_4 = model_four()

model_1.save('my_model.h5')
"""
show output
"""
acc = model_1.history['acc']
val_acc_1 = model_1.history['val_acc']
# val_acc_2 = model_2.history['val_acc']
# val_acc_3 = model_3.history['val_acc']
# val_acc_4 = model_4.history['val_acc']
loss = model_1.history['loss']
val_loss = model_1.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_1, 'b', label='Validation Acc 1')
# plt.plot(epochs, val_acc_2, 'r', label='Validation Acc 2')
# plt.plot(epochs, val_acc_2, 'g', label='Validation Acc 3')
# plt.plot(epochs, val_acc_4, 'y', label='Validation Acc 4')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.legend()

plt.show()


"""
Evaluate
"""
test_loss, test_acc = model_1.evaluate(test_texts, test_labels)
print(test_acc)