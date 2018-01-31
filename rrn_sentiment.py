from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from random import shuffle

import json

import random

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

def pre_process(s):
	tokens=nltk.word_tokenize(s)
	import string
	punctuations = set(list(string.punctuation))
	removed_punctuation=[j for j in tokens if j not in punctuations]
	case_folding=[]
	for j in removed_punctuation:
		case_folding.append(j.lower())
	stop = set(stopwords.words('english'))
	stopword_removal=[j for j in case_folding if j not in stop]
	stemmer = PorterStemmer()
	stem=[stemmer.stem(words) for words in stopword_removal]
	result = ' '.join(stem)
	return result

max_features = 100000
maxlen = 250  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')

with open("neg_amazon_cell_phone_reviews.json") as jf:
	jdn = json.load(jf)

with open("pos_amazon_cell_phone_reviews.json") as jf:
	jdp = json.load(jf)

n_summaries = []
p_summaries = []
n_ratings = []
p_ratings = []

max_no = 3000

for i in range(max_no):
	temp = pre_process(jdp["root"][i]['text'])
	st = jdp["root"][i]['summary'] + ' NOTIMPORTANT ' + temp[:min(max_features - 20000, len(temp))]
	p_summaries.append(st)
	p_ratings.append(1)

for i in range(max_no):
	temp = pre_process(jdn["root"][i]['text'])
	st = jdn["root"][i]['summary'] + ' NOTIMPORTANT ' + temp[:min(max_features - 20000, len(temp))]
	n_summaries.append(st)
	#n_ratings.append(int(jdn["root"][i]['rating']))
	n_ratings.append(0)

summaries = n_summaries + p_summaries
ratings = n_ratings + p_ratings

z = zip(summaries, ratings)
shuffle(z)
summaries[:], ratings[:] = zip(*z)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(n_summaries)

x_train = tokenizer.texts_to_sequences(p_summaries[0:max_no] + n_summaries[0:max_no])
x_test = tokenizer.texts_to_sequences(summaries[0:250])
y_train = p_ratings[0:max_no] + n_ratings[0:max_no]
y_test = ratings[0:250]

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))

# try using different optimizers and different optimizer configs(category_crossentropy can be used either)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test loss:', loss)	#in decimal
print('Test accuracy:', acc)	#in decimal
