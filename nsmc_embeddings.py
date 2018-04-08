PATH = "./nsmc/"
ENTIRE_FILE = 'ratings.txt'
TRAIN_FILE = "ratings_train.txt"
TEST_FILE = "ratings_test.txt"
MODE = "SYLLABLE"
MODEL = "fastText"

# MAX_SEQUENCE_LENGTH = 50
# EMBEDDING_DIM = 300

from tqdm import tqdm
import re
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

import keras
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from konlpy.tag import Twitter
import hgtk
from kor_romanize import Romanizer
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
import multiprocessing
from glove import Corpus, Glove


from pathlib import Path

def read_data(filename):
    with open(filename, 'rt', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

# save glove vectors to word2vec format
def save_word2vec_format(glove, filename):
    with open(filename, 'w') as savefile:
        (rows, cols) = glove.word_vectors.shape
        savefile.write(str(rows) + " " + str(cols) + "\n")
        if hasattr(glove.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = glove.dictionary.iteritems()
        else:
            items_iterator = glove.dictionary.items()

        for word, idx in items_iterator:
            vector = glove.word_vectors[idx]
            vector_string = ""
            for val_i in vector:
                vector_string += " " + str(val_i)
            savefile.write((word + vector_string + "\n"))




''' glove paramaters
        - int no_components: number of latent dimensions
        - float learning_rate: learning rate for SGD estimation.
        - float alpha, float max_count: parameters for the
          weighting function (see the paper).
        - float max_loss: the maximum absolute value of calculated
                          gradient for any single co-occurrence pair.
                          Only try setting to a lower value if you
                          are experiencing problems with numerical
                          stability.
        - random_state: random statue used to intialize optimization
        """
'''


def create_word_embddings(tokens, model_type, params, file_suffix):
    if(model_type == "word2vec"):
        model = Word2Vec(tokens, **params)
        word_vectors = model.wv
        word_vectors.save_word2vec_format("./embeddings/"+MODE+"_"+model_type+"_nsmc_"+file_suffix)
    elif(model_type =="fastText"):
        model = FastText(tokens, **params)
        word_vectors = model.wv
        word_vectors.save_word2vec_format("./embeddings/"+MODE+"_"+model_type+"_nsmc_"+file_suffix)
    elif(model_type =="glove"):
        nb_components = params['size']
        no_threads = params['workers']
        window = params['window']
        corpus = Corpus()
        corpus.fit(tokens, window=window)
        glove = Glove(no_components=nb_components, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs=5, no_threads=no_threads, verbose=True)
        save_word2vec_format(glove, "./embeddings/"+MODE+"_"+model_type+"_nsmc_"+file_suffix)
    else:
        raise ValueError

def create_model(word_index, embed_dim, max_sequence_length, embedding_matrix):
    from keras.layers import Embedding
    embedding_layer = Embedding(len(word_index) + 1,
                                embed_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)


    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
    preds = Dense(2, activation='sigmoid')(lstm)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def load_word_vectors(fname):
    word_vectors = KeyedVectors.load_word2vec_format(fname)  # C text format
    return word_vectors

def compute_embedding_matrix(word_vectors, embedding_dimension, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors.get_vector(word) # fixed
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector[:embedding_dimension]
        except:
            continue
    return embedding_matrix


def tokenize_words(sentence):
    return re.findall('\w+', sentence)

def tokenize_morpheme(doc):
  # norm, stem은 optional
  # return ['/'.join(t) for t in twitter.pos(doc, norm=True, stem=True)]
  return [t for t in twitter.morphs(doc, norm=True, stem=True)]

def word_to_jamo_seqs(word):
    temp = []
    for letter in word:
        try:
            temp.extend(hgtk.letter.decompose(letter))
        except:
            continue
    return "".join(temp)

def process_text(unit, train_data):
    texts = [] # list of sentences
    tokens = [] # list of words
    if (unit == "MORPHEME"):
        for i in tqdm(range(1, len(train_data))):
            token = tokenize_morpheme(train_data[i][1])
            processed_sentence = " ".join(token)
            tokens.append(token)
            texts.append(processed_sentence)
    elif (unit == "WORD" or unit == "SYLLABLE"):
        for i in tqdm(range(1, len(train_data))):
            token = tokenize_words(train_data[i][1])
            processed_sentence = " ".join(token)
            tokens.append(token)
            texts.append(processed_sentence)
    elif (unit == "JAMO"):
        for i in tqdm(range(1, len(train_data))):
            token = tokenize_words(train_data[i][1])
            jamo_token = []
            for word in token:
                jamo_token.append(word_to_jamo_seqs(word))
            processed_sentence = " ".join(jamo_token)
            tokens.append(jamo_token)
            texts.append(processed_sentence)
    elif (unit == "ROMANIZE"):
        for i in tqdm(range(1, len(train_data))):
            token = [Romanizer(t).romanize() for t in tokenize_words(train_data[i][1])]
            processed_sentence = " ".join(token)
            tokens.append(token)
            texts.append(processed_sentence)
    else:
        import sys
        sys.exit()
    return tokens, texts

def create_data_x(texts):
    tokenizer = Tokenizer(filters="")  # do not designate filters
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data_x = pad_sequences(sequences) # maybe set to 50
    return data_x, word_index

def make_parmas(param_options):
    params_list = []
    for size in param_options['size']:
        for window in param_options['size']:
            for min_count in param_options['min_count']:
                for workers in param_options['workers']:
                    for sample in param_options['sample']:
                        for iteration in param_options['iter']:
                            param = {
                                'size': size,
                                'window': window,
                                'min_count': min_count,
                                'workers': workers,
                                'sample': sample,
                                'iter': iteration}
                            params_list.append(param)
    return params_list

def visualize_result(history, fname):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fname+'_accuracy.png', bbox_inches='tight')
    plt.close()
    time.sleep(3)


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(block=False)
    plt.savefig(fname + '_accuracy.png', bbox_inches='tight')
    plt.close()
def main():
    train_data = read_data(PATH + ENTIRE_FILE)
    saved_tokens = Path("./preprocessed/"+MODE + "_tokens")
    saved_texts = Path("./preprocessed/"+MODE + "_texts")
    if saved_tokens.is_file() and saved_texts.is_file():
        tokens = pickle.load(open("./preprocessed/"+MODE+"_tokens", "rb"))
        data_x = pickle.load(open("./training_samples/"+MODE+"_data_x", "rb"))
        max_sequence_length = data_x.shape[1]
        print("max_sequence_length: ",max_sequence_length)
        data_y = pickle.load(open("./training_samples/"+MODE+"_data_y", "rb"))
        word_index = pickle.load(open("./training_samples/"+MODE+"_word_index", "rb"))
        # texts = pickle.load(open(MODE+"_texts", "rb"))
    else:
        tokens, texts = process_text(unit=MODE, train_data=train_data)
        with open("./preprocessed/"+MODE + "_tokens", "wb") as f:
            pickle.dump(tokens, f)
        with open("./preprocessed/"+MODE + "_texts", "wb") as f:
            pickle.dump(texts, f)
        # prepare data x
        data_x, word_index = create_data_x(texts)
        max_sequence_length = data_x.shape[1]
        # prepare data y
        y_labels = [row[2] for row in train_data[1:]]  # positive 1, negative 0
        data_y = np_utils.to_categorical(np.asarray(y_labels))
        with open("./training_samples/"+MODE + "_data_x", "wb") as f:
            pickle.dump(data_x, f)
        with open("./training_samples/"+MODE + "_data_y", "wb") as f:
            pickle.dump(data_y, f)
        with open("./training_samples/"+MODE + "_word_index", "wb") as f:
            pickle.dump(word_index, f)

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    param_options = {
        #'size':[50],
        'size':[50, 100, 300, 500, 1000],
        #'window':[5],
        'window':[2,5,7,10],
        #'min_count':[10],
        'min_count':[10, 20, 50, 100],
        'workers':[max_workers],
        'sample':[1E-3],
        'iter':[5]
    }
    params_list = make_parmas(param_options) # 100 different parameters

    # params = {'size': 300, 'window': 5, 'min_count': 4,
    #           'workers': max(1, multiprocessing.cpu_count() - 1), 'sample': 1E-3}  # 'iter':5

    for params in params_list:
        file_suffix = str(list(params.values()))
        print("running : "+file_suffix)
        create_word_embddings(tokens=tokens, model_type=MODEL, params=params, file_suffix=file_suffix)
        # compute embedding matrix
        word_vectors = load_word_vectors("./embeddings/"+MODE+"_"+MODEL+"_nsmc_"+file_suffix)
        embedding_matrix = compute_embedding_matrix(word_vectors=word_vectors, embedding_dimension=params['size'], word_index=word_index)
        # define network
        model = create_model(word_index, params['size'],max_sequence_length, embedding_matrix)
        # train test split
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, shuffle=False, test_size=0.25)

        file_suffix = str(list(params.values()))
        filepath = "./model/{0}_{1}_{2}.hdf5".format(MODE, MODEL, file_suffix)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='auto')
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')
        callbacks_list = [checkpoint, early_stopping]

        history = model.fit(x_train,
                  y_train,
                  shuffle=True,
                  epochs=100,
                  batch_size=32,
                  validation_data=(x_test, y_test),
                  callbacks=callbacks_list,
                  verbose=1)
        with open('./history/'+MODE+"_"+MODEL+"_nsmc_"+file_suffix, 'wb') as f:
            pickle.dump(history.history, f)
        visualize_result(history, fname=file_suffix)



if __name__ == "__main__":
    # model = keras.models.load_model('./model/' +"WORD_300"  + ".hdf5")
    # print(model.summary())
    twitter = Twitter()
    main()
