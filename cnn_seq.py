import nltk
import numpy as np
from w2v import train_word2vec
import sys
sys.path.insert(0, '/path/to/nlp_scripts/')
from file_ops import open_csv
from random import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence

np.random.seed(0)

embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

batch_size = 64
num_epochs = 10
sequence_length = # to be calculated based on the largest document
max_words = 5000

min_word_count = 1
context = 10

data = open_csv('csv_file.csv')

def convert_sent_to_ind(sent):
    sent = nltk.word_tokenize(sent)
    inds = [vocabulary_inv_rev[word] for word in sent]
    while len(inds) < sequence_length:
        inds.append(0)
    return inds

def convert_word_to_index(rows, inv):
    rows_out = []
    counter = 0
    for row in rows:
        try:
            conv_sent = []
            row[0] = nltk.word_tokenize(row[0])
            if len(row[0]) > counter:
                counter = len(row[0])
            for word in row[0]:
                ind = inv[word]
                conv_sent.append(ind)
        except:
            continue
        while len(conv_sent) < sequence_length:
            conv_sent.append(0)
        conv_sent = np.array(conv_sent)
        row_out = [conv_sent, row[1]]
        rows_out.append(row_out)
    return rows_out

def replace(data, label_list):
    for ind, label in enumerate(label_list):
        for row in data:
            if row[1] == label:
                row[1] = ind
    return data

def load_my_data():
    shuffle(data)
    sents = [row[0] for row in data]

    all_tokens = []
    for row in data:
        tokens = nltk.word_tokenize(row[0])
        all_tokens.extend(tokens)
    tokens_set = set(all_tokens)

    vocabulary_inv = dict((i+1, j) for i, j in enumerate(tokens_set))
    vocabulary_inv_rev = dict((j, i+1) for i, j in enumerate(tokens_set))
    vocabulary_inv[0] = "<PAD/>"
    vocabulary_inv_rev["<PAD/>"] = 0

    converted_sents = convert_word_to_index(data, vocabulary_inv_rev)
    x_train = np.array([row[0] for row in converted_sents[0:100000]])
    x_test = np.array([row[0] for row in converted_sents[100000:-1]])
    y_train = np.array([int(row[1]) for row in converted_sents[0:100000]])
    y_test = np.array([int(row[1]) for row in converted_sents[100000:-1]])
    return x_train, y_train, x_test, y_test, vocabulary_inv, vocabulary_inv_rev

def evaluate_model():
    try:
        print(model.evaluate(x_train, y_train, batch_size=batch_size, verbose=2))
        loss, acc = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=2)
        print(loss, acc)
    except Exception as e:
        print(e)
    return

def predict_all(data):
    sents = [row[0] for row in data]
    converted = []
    for s in sents[10:30]:
        try:
            stimulus_inds = convert_sent_to_ind(s)
            stimulus_inds = np.array(stimulus_inds)
            converted.append(stimulus_inds)
        except Exception as e:
            print(e)
    converted = np.array(converted)
    predictions = model.predict(converted)
    print(predictions)
    return

x_train, y_train, x_test, y_test, vocabulary_inv, vocabulary_inv_rev = load_my_data()

embedding_weights = train_word2vec(np.vstack((x_train, x_test)),
                    vocabulary_inv, num_features=embedding_dim,
                    min_word_count=min_word_count, context=context)
                    
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)
z = Embedding(len(vocabulary_inv), embedding_dim,
                input_length=sequence_length, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)

conv_blocks = []
for size in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=size,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks)

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

weights = np.array([v for v in embedding_weights.values()])
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)
evaluate_model()
predict_all(testing_data)
