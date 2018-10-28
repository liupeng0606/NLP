import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, GlobalAveragePooling1D
from keras.layers import Embedding, Conv2D, GlobalMaxPooling1D
from keras import regularizers
from sklearn.model_selection import train_test_split
nltk.download('stopwords')


df = pd.read_csv("labeledTrainData.tsv", sep = '\t',
                 error_bad_lines=False )
print("Total no. of reviews are ", df.shape[0])
print("cols are ", df.columns)
print("Sample reviews are ")
print(df.loc[:5,['review','sentiment']])


corpus = []
s = set(stopwords.words('english'))
s.remove('not')
print("Stopwords length", len(s))

word2vec = {}
with open('glove.6B.50d.txt', encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))


MAX_VCOCAB_SIZE = 5000
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 1500
tokenizer = Tokenizer( filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
sequences = tokenizer.fit_on_texts(df['review'])
word_index = tokenizer.word_index
documents = tokenizer.texts_to_sequences(df['review'])
#print(word_index)
token_count = len(word_index)+1
print('Found {} unique tokens.'.format(token_count))

#print(t.word_counts)
print("Total documents ", tokenizer.document_count)
#print(t.word_index)
#print(t.word_docs)
print("max sequence length:", max(len(s) for s in documents))
print("min sequence length:", min(len(s) for s in documents))

# pad sequences so that we get a N x T matrix
data = pad_sequences(documents, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print('Shape of data tensor:', data.shape)
print(data[1])

print('Filling pre-trained embeddings...')
embedding_matrix = np.zeros((token_count, EMBEDDING_DIM))
for word, i in word_index.items():
  #if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word) #get(word) is used instead of [word] as it won't give exception in case word is not found
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i,:] = embedding_vector

print("Sample embedded dimension ")
print(embedding_matrix[10][:5])



embedding_layer = Embedding(
  token_count,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False)

model = Sequential()
model.add(embedding_layer)#, input_shape= (token_count, EMBEDDING_DIM))
model.add(Conv1D(filters = 64, kernel_size = 4, padding = 'same', activation='relu'))
#input_shape=(token_count,EMBEDDING_DIM)))
model.add(MaxPooling1D())#kernel_size=500))
model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'same',  activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.25))
model.add(MaxPooling1D())
model.add(Conv1D(filters = 256, kernel_size = 2, padding = 'same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
#model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(data, df['sentiment'],
                                                    test_size=0.2, random_state=42)
model.fit(x_train, y_train , batch_size=96, epochs=35, validation_split = 0.25)

score = model.evaluate(x_test, y_test, batch_size=32)
print(model.summary())



