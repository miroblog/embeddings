# encoding: utf-8

from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import re
import os

def load_word_vectors(fname):
    word_vectors = KeyedVectors.load_word2vec_format(fname)  # C text format
    return word_vectors

def parse_file_name(file_name):
    tokens = re.split("[_,]", file_name)
    param = {}
    for i, token in enumerate(tokens):
        token = token.strip()
        if (i == 0):
            param['type'] = token
        if (i == 1):
            param['model'] = token
        if("=" in token):
            name, value = token.split("=")
            param[name] = value
    return param

def file_selector(path, param):
    files = os.listdir(path)
    for file in files:
        extracted_param = parse_file_name(file)
        print(extracted_param)
        if(param['model'] == extracted_param['model'] and
            param['type'] == extracted_param['type'] and
            param['window'] == int(extracted_param['window']) and
            param['size'] == int(extracted_param['size']) and
            param['count'] == int(extracted_param['count'])):
            return file
        else:
            continue
    return None

# model_path = "MORPHEME_fastText_nsmc_sample=0.001, workers=11, parser='twitter', iter=5, window=5, size=300, min_count=20"
def create_embedding_projector(path, file_name):
    param = parse_file_name(file_name)
    DIM = int(param['size'])
    NAME = "projector_{0}_{1}_{2}_{3}_{4}".format(param['type'],
                                              param['model'],
                                              int(param['size']),
                                            int(param['window']),
                                              int(param['count']))
    PATH = "./"+NAME+"/"
    try:
        os.mkdir(NAME)
    except:
        print("path already exists")
    wv = load_word_vectors(path+file_name)
    w2v = np.zeros((len(wv.index2word), DIM))
    with open(PATH + "metadata.tsv", 'wb') as file_metadata:
        for i, word in enumerate(wv.index2word):
            w2v[i] = wv[word]
            file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')


    # setup a TensorFlow session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=[None, DIM])
    set_x = tf.assign(X, place, validate_shape=False)

    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: w2v})

    # create a TensorFlow summary writer
    summary_writer = tf.summary.FileWriter(NAME, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = PATH + 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, PATH + 'model.ckpt')
    sess.close()

    print("to run, type 'tensorboard --logdir={0}".format(PATH))