from tensorflow.python.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import re

in_filename_test = "test.ft.txt"
# in_filename = "head.txt"
te = open(in_filename_test)
test_file_lines = te.readlines()
te.close()

model = load_model('my_model.h5')

lines = ['This was the best tool I have ever used. I would recommend it to all my friends!',
         'It arrived broken and customer support took forever.',
         "Great for the non-audiophile: Reviewed quite a bit of the combo players and was hesitant due to unfavorable reviews and size of machines. I am weaning off my VHS collection, but don't want to replace them with DVD's. This unit is well built, easy to setup and resolution and special effects (no progressive scan for HDTV owners) suitable for many people looking for a versatile product.Cons- No universal remote."]
labels = [1, 0, 1]
texts = [x.split(' ', 1)[1][:-1].lower() for x in lines]

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
test_texts = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

test_labels = test_labels[:int(len(test_labels)/2)]
test_texts = test_texts[:int(len(test_texts)/2)]

# convert digits to 0
for i in range(len(test_texts)):
    test_texts[i] = re.sub('\d', '0', test_texts[i])

# convert urls to "<url>"
for i in range(len(test_texts)):
    if 'www.' in test_texts[i] or 'http:' in test_texts[i] or 'https:' in test_texts[i] or '.com' in \
            test_texts[i]:
        test_texts[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_texts[i])


print('loading data...')

max_len = 100
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(test_texts)
sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.index_word
data = pad_sequences(sequences, maxlen=max_len)
label = np.asarray(test_labels)

print(model.evaluate(data, label, batch_size=4000))

pred = model.predict(data[:10])
print('prediction:', pred)