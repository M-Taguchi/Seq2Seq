# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import random
import numpy.random as nr
import sys
import math
import time
import pickle
import gc
import os

from keras.layers.core import Dense
from keras.layers.core import Masking
from keras.layers import Input
from keras.models import Model
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model

from pyknp import Jumanpp
import codecs
import argparse

from model import *

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="2", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

#辞書をロード
with open("words_indices.pickle", "rb") as l :
    words_indices = pickle.load(l)

with open("indices_words.pickle", "rb") as m :
    indices_words = pickle.load(m)

#単語ファイルロード
with open("words.pickle", "rb") as ff :
    words = pickle.load(ff)

#Encoder Inputデータをロード
with open("e.pickle", "rb") as f :
    e = pickle.load(f)

#Decoder Inputデータをロード
with open("d.pickle", "rb") as g :
    d = pickle.load(g)

#ラベルデータをロード
with open("t.pickle", "rb") as h :
    t = pickle.load(h)

#maxlenロード
with open("maxlen.pickle", "rb") as maxlen :
    [maxlen_e, maxlen_d] = pickle.load(maxlen)

#訓練データとテストデータを95:5に分割
n_split = int(e.shape[0]*0.95)
e_train, e_test = np.vsplit(e,[n_split])
d_train, d_test = np.vsplit(d,[n_split])
t_train, t_test = np.vsplit(t,[n_split])
vec_dim = 400
epochs = 10
batch_size = 32
input_dim = len(words)
output_dim = input_dim
#隠れ層の次元
n_hidden = int(vec_dim*2)

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="optional", action="store_true")
args = parser.parse_args()

if args.train :
	prediction = Seq2SeqModel(maxlen_e,maxlen_d,n_hidden,input_dim,vec_dim,output_dim)
	emb_param = "param_seq2seq021.hdf5"
	row = e_train.shape[0]
	e_train = e_train.reshape(row,maxlen_e)
	d_train = d_train.reshape(row,maxlen_d)
	t_train = t_train.reshape(row,maxlen_d)
	model = prediction.train(e_train,d_train,t_train,batch_size,epochs,emb_param)
	#ネットワーク図出力
	plot_model(model, show_shapes=True,to_file='seq2seq021.png')
	#学習済みパラメータセーブ
	model.save_weights(emb_param)                               

	row2 = e_test.shape[0]
	e_test = e_test.reshape(row2,maxlen_e)
	d_test = d_test.reshape(row2,maxlen_d)
	#t_test=t_test.reshape(row2,maxlen_d)
	print()
	perplexity = prediction.eval_perplexity(model,e_test,d_test,t_test,batch_size) 
	print('Perplexity=',perplexity)

else :
    dialog = Seq2SeqModel(maxlen_e,1,n_hidden,input_dim,vec_dim,output_dim)
    model, encoder_model, decoder_model = dialog.create_model()
	
    plot_model(encoder_model,show_shapes=True,to_file="seq2seq021_encoder.png")
    plot_model(decoder_model,show_shapes=True,to_file="seq2seq021_decoder.png")
    emb_param = "param_seq2seq021.hdf5"
    model.load_weights(emb_param)
    sys.stdin = codecs.getreader("utf_8")(sys.stdin)

    jumanpp = Jumanpp()

    while True :
        cns_input = input(">> ")
        if cns_input == "q":
            print("終了")
            break

        result = jumanpp.analysis(cns_input)
        input_text = []
        for mrph in result.mrph_list():
            input_text.append(mrph.midasi)

        mat_input = np.array(input_text)

        #入力データe_inputに入力文の単語インデックスを設定
        e_input = np.zeros((1,maxlen_e))
        for i in range(0,len(mat_input)) :
            if mat_input[i] in words :
                e_input[0,i] = words_indices[mat_input[i]]
            else :
                e_input[0,i] = words_indices["UNK"]

        input_sentence=''
        for i in range(0,maxlen_e) :
            j = e_input[0,i]
            if j != 0 :
                input_sentence +=indices_words[j]
            else :
                break

        #応答文組み立て
        response = dialog.response(encoder_model,decoder_model,e_input,maxlen_d,words_indices,indices_words)

        print(response)
