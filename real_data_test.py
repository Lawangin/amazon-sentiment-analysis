from tensorflow.python.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

model = load_model('my_model.h5')

lines = ['This was the best tool I have ever used. I would recommend it to all my friends!',
         'It arrived broken and customer support took forever.']
texts = [x.split(' ', 1)[1][:-1].lower() for x in lines]

print('loading data...')

max_len = 100
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.index_word
data = pad_sequences(sequences, maxlen=max_len)

pred = model.predict(data)
print('prediction:', pred)