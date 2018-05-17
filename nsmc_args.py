# -*- coding: utf-8 -*-

from tqdm import tqdm
import nltk
import re
import pickle

# keras
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K

# sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf

# konlpy
from konlpy.tag import Twitter, Kkma, Komoran
import hgtk
from kor_romanize import Romanizer

# embedding models
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from glove import Corpus, Glove

# os
import multiprocessing
from pathlib import Path


def read_data(filename):
    with open(filename, 'rt', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data


# save glove vectors to word2vec format
def save_word2vec_format(glove, filename):
    with open(filename, 'w', encoding='utf-8') as savefile:
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

def create_word_embeddings(tokens, model_type, params, file_suffix):
    if (model_type == "word2vec"):
        params['sg'] = UseSkipGram # 1 : skipgram, 0 : cbow
        model = Word2Vec(tokens, **params)
        word_vectors = model.wv
        word_vectors.save_word2vec_format("./embeddings/" + MODE + "_" + model_type + "_nsmc_" + file_suffix)
    elif (model_type == "fastText"):
        params['min_n'] = 1
        params['sg'] = UseSkipGram
        model = FastText(tokens, **params)
        word_vectors = model.wv
        word_vectors.save_word2vec_format("./embeddings/" + MODE + "_" + model_type + "_nsmc_" + file_suffix)
    elif (model_type == "glove"):
        dict = restrict_rare_words(params["min_count"], tokens)
        nb_components = params['size']
        no_threads = params['workers']
        window = params['window']
        corpus = Corpus(dictionary=dict)
        corpus.fit(tokens, window=window, ignore_missing=True)
        glove = Glove(no_components=nb_components, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs=5, no_threads=no_threads, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        save_word2vec_format(glove, "./embeddings/" + MODE + "_" + model_type + "_nsmc_" + file_suffix)
    else:
        raise ValueError


# returns dictionary
def restrict_rare_words(min_count, tokens):
    flattend_list = [j for i in tokens for j in i]
    freq_dist = nltk.FreqDist(flattend_list)
    common_word = {}
    i = 0
    for word, count in freq_dist.most_common()[-1::-1]:
        if (count > min_count):
            common_word[word] = i
            i += 1
    return common_word


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
            embedding_vector = word_vectors.get_vector(word)  # fixed
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector[:embedding_dimension]
        except:
            continue
    return embedding_matrix


def tokenize_words(sentence):
    return re.findall('\w+', sentence)


def tokenize_morpheme(doc, parser):
    return [t for t in parser.morphs(doc)]


def word_to_jamo_seqs(word):
    temp = []
    for letter in word:
        try:
            temp.extend(hgtk.letter.decompose(letter))
        except:
            continue
    return "".join(temp)


def process_text(unit, train_data, parser):
    texts = []  # 1d - list of sentences
    tokens = []  # 2d - sentence - unit(word/morpheme)
    if (unit == "MORPHEME" or "SYLLABLE+MORPHEME"):
        for i in tqdm(range(1, len(train_data))):
            token = tokenize_morpheme(train_data[i][1], parser)
            processed_sentence = " ".join(token)
            tokens.append(token)
            texts.append(processed_sentence)
    elif (unit == "WORD" or unit == "SYLLABLE"):
        for i in tqdm(range(1, len(train_data))):
            token = tokenize_words(train_data[i][1])
            processed_sentence = " ".join(token)
            tokens.append(token)
            texts.append(processed_sentence)
    elif(unit =="JAMO+MORPHEME"):
        for i in tqdm(range(1, len(train_data))):
            token = tokenize_morpheme(train_data[i][1], parser)
            jamo_token = []
            for word in token:
                jamo_token.append(word_to_jamo_seqs(word))
            processed_sentence = " ".join(jamo_token)
            tokens.append(jamo_token)
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
    data_x = pad_sequences(sequences)  # maybe set to 50
    return data_x, word_index


def make_parmas(param_options):
    params_list = []
    for size in param_options['size']:
        for window in param_options['window']:
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


'''
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
    plt.savefig(fname + '_loss.png', bbox_inches='tight')
    plt.close()
'''


def make_file_suffix(dict):
    temp_dict = dict.copy()
    temp_dict['parser'] = args.parser
    suffix = ', '.join("{!s}={!r}".format(k, v) for (k, v) in temp_dict.items())
    return suffix


def train_sentiment(params, tokens, word_index, max_sequence_length, data_x, data_y, verbose=False):
    file_suffix = make_file_suffix(params)
    if(verbose):
        print("running : " + file_suffix)
    create_word_embeddings(tokens=tokens, model_type=MODEL, params=params, file_suffix=file_suffix)
    # compute embedding matrix
    word_vectors = load_word_vectors("./embeddings/" + MODE + "_" + MODEL + "_nsmc_" + file_suffix)
    embedding_matrix = compute_embedding_matrix(word_vectors=word_vectors, embedding_dimension=params['size'],
                                                word_index=word_index)

    # memory issues
    K.clear_session()
    sess = tf.Session()
    K.set_session(sess)

    # define network
    model = create_model(word_index, params['size'], max_sequence_length, embedding_matrix)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, shuffle=False, test_size=0.25)

    filepath = "./model/{0}_{1}_{2}.hdf5".format(MODE, MODEL, file_suffix)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', patience=PATIENCE, verbose=1, mode='auto')
    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(x_train,
                        y_train,
                        shuffle=True,
                        epochs=MAX_EPOCH,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list,
                        verbose=1)
    with open('./'+ PATH_OUT + "/" + MODE + "_" + MODEL + "_nsmc_" + file_suffix, 'wb') as f:
        pickle.dump(history.history, f)
    return max(history.history['val_acc'])

def main():
    # read data by sentences
    train_data = read_data(PATH + ENTIRE_FILE)
    mode_affix = args.parser + "_" + MODE

    # check for preprocessed text
    saved_tokens = Path(PATH_PREPROCESSED + mode_affix+ "_tokens")
    saved_texts = Path(PATH_PREPROCESSED + mode_affix+ "_texts")
    # load, if already exists
    if saved_tokens.is_file() and saved_texts.is_file():
        tokens = pickle.load(open(PATH_PREPROCESSED + mode_affix+"_tokens", "rb"))
        data_x = pickle.load(open(PATH_TRAINING_SAMPLES + mode_affix+"_data_x", "rb"))
        max_sequence_length = data_x.shape[1]
        print("max_sequence_length: ", max_sequence_length)
        data_y = pickle.load(open(PATH_TRAINING_SAMPLES + mode_affix+ "_data_y", "rb"))
        word_index = pickle.load(open(PATH_TRAINING_SAMPLES + mode_affix+"_word_index", "rb"))
    # make, if isn't processed
    else:
        tokens, texts = process_text(unit=MODE, train_data=train_data, parser=parser)
        with open(PATH_PREPROCESSED +mode_affix+ "_tokens", "wb") as f:
            pickle.dump(tokens, f)
        with open(PATH_PREPROCESSED +mode_affix+ "_texts", "wb") as f:
            pickle.dump(texts, f)

        # prepare data x
        data_x, word_index = create_data_x(texts)
        max_sequence_length = data_x.shape[1]

        # prepare data y
        y_labels = [row[2] for row in train_data[1:]]  # positive 1, negative 0
        data_y = np_utils.to_categorical(np.asarray(y_labels))

        # dump for later usage
        with open(PATH_TRAINING_SAMPLES +mode_affix+ "_data_x", "wb") as f:
            pickle.dump(data_x, f)
        with open(PATH_TRAINING_SAMPLES +mode_affix+ "_data_y", "wb") as f:
            pickle.dump(data_y, f)
        with open(PATH_TRAINING_SAMPLES +mode_affix+ "_word_index", "wb") as f:
            pickle.dump(word_index, f)

    # make parameters
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    param_options = {
        'size': [50, 100, 300, 500, 1000],
        'window': [2, 5, 7, 10],
        'min_count': [10, 20, 50, 100],
        'workers': [max_workers],
        'sample': [1E-3],
        'iter': [5]
    }
    params_list = make_parmas(param_options)

    # run all parameter combinations
    for params in params_list:
        train_sentiment(params, tokens, word_index, max_sequence_length, data_x, data_y)

# takes list of directory name
def make_directories(directories):
    for directory in directories:
        try:
            os.mkdir(directory)
        except:
            print("Path already exists :", directory)
    print("made following directories : "+str(directories))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parser", default="twitter", help="morpheme parser: kkma, komoran, twitter")
    parser.add_argument("-m", "--model", help="model name : word2vec, fastText, glove")
    parser.add_argument("-u", "--unit", help="unit : WORD, MORPHEME, SYLLABLE, JAMO")
    parser.add_argument("-o", "--output_path", help="output path")
    parser.add_argument("-v", "--variation", default='sg', help="variation : sg, cbow")
    parser.add_argument("-min", "--min_ngram", default=1, help="minimum n gram: 1~3")
    args = parser.parse_args()
    MODE = args.unit #"MORPHEME"
    MODEL = args.model #"fastText"
    Minimum_N_Gram = args.min_ngram
    UseSkipGram = 1 if args.variation == "sg" else 0 # "sg"

    print(str(args))
    if(args.parser == "kkma"):
        parser = Kkma()
    elif(args.parser == "komoran"):
        parser = Komoran()
    else:
        parser = Twitter()

    # Data
    PATH = "./nsmc/"
    ENTIRE_FILE = 'ratings.txt'
    TRAIN_FILE = "ratings_train.txt"
    TEST_FILE = "ratings_test.txt"

    # Processed text
    PATH_PREPROCESSED = "./preprocessed/"
    # Training samples
    PATH_TRAINING_SAMPLES = "./training_samples/"

    # Embeddings PATH, trained gensim model in w2vec format
    PATH_EMBEDDING = "./embeddings/"

    # Output PATH
    PATH_OUT = args.output_path

    PATH_MODEL = "./model/"

    # Network Training Params
    BATCH_SIZE = 128
    MAX_EPOCH = 20
    PATIENCE = 5

    make_directories([
        PATH_PREPROCESSED, PATH_TRAINING_SAMPLES, PATH_EMBEDDING, PATH_OUT, PATH_MODEL
    ])
    main()
