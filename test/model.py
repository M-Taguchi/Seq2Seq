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

from __future__ import print_function
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

class Seq2SeqModel :
    def __init__(self,max_len_e,max_len_d,n_hidden,input_dim,vec_dim,output_dim) :
        self.maxlen_e = maxlen_e
        self.maxlen_d = maxlen_d
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.vec_dim = vec_dim
        self.output_dim = output_dim

    def create_model(self) :
        print("#2")
        #エンコーダ
        encoder_input = Input(shape=(self.maxlen_e,),dtype="int16",name="encoder_input")
        e_i = Embedding(output_dim=self.vec_dim,input_dim=self.input_dim,#input_length=self.maxlen_e,
                       mask_zero=True, embeddings_initializer=uniform(seed=20190415))(encoder_input)
        e_i = BatchNormalization(axis=-1)(e_i)
        e_i = Masking(mask_value=0.0)(e_i)
        encoder_outputs, state_h, state_c = LSTM(self.n_hidden,name="encoder_LSTM",
                                                return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20190415),
                                                recurrent_initializer=orthogonal(gain=1.0,seed=20190415),
                                                dropout=0.5,recurrent_dropout=0.5)(e_i)

        #'encoder_outputs'は使わない。statesのみ使用する
        encoder_states = [state_h, state_c]

        encoder_model = Model(inputs=encoder_input,outputs=[encoder_outputs,state_h,state_c])

        print("#3")
        #デコーダ(学習用)
        #レイヤ定義
        decoder_LSTM = LSTM(self.n_hidden,name="decoder_LSTM",
                           return_sequences=True,return_state=True,
                           kernel_initializer=glorot_uniform(seed=20190415),
                           recurrent_initializer=orthogonal(gain=1.0,seed=20190415),
                           dropout=0.5,recurrent_dropout=0.5)
        decoder_Dense = Dense(self.output_dim,activation="softmax",name="decoder_Dense",
                             kernel_initializer=glorot_uniform(seed=20190415))
        #入力
        decoder_inputs = Input(shape=(self.maxlen_d,),dtype="int16",name="decoder_inputs")
        d_i = Embedding(output_dim=self.vec_dim,input_dim=self.input_dim,#input_length=self.maxlen_d,
                       mask_zero=True,embeddings_initializer=uniform(seed=20190415))(decoder_inputs)
        d_i = BatchNormalization(axis=-1)(d_i)
        d_i = Masking(mask_value=0.0)(d_i)
        d_input = d_i
        #LSTM
        d_outputs, _, _ = decoder_LSTM(d_i,initial_state=encoder_states)
        print("#4")
        decoder_outputs = decoder_Dense(d_outputs)
        model = Model(inputs=[encoder_input,decoder_inputs],outputs=decoder_outputs)
        model.compile(loss="categorical_cross_entropy",optimizer="Adam",metrics=["categorical_accuracy"])

        #デコーダ(応答文生成)
        print("#5")
        #入力定義
        decoder_state_input_h = Input(shape=(self.n_hidden,),name="input_h")
        decoder_state_input_c = Input(shape=(self.n_hidden,),name="input_c")
        decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]
        #LSTM
        decoder_lstm, state_h, state_c = decoder_LSTM(d_input,initial_state=decoder_states_inputs)
        decoder_states = [state_h,state_c]
        decorder_res = decoder_Dense(decoder_lstm)
        decoder_model = Model([decoder_inputs]+decoder_states_inputs,[decorder_res]+decoder_states)

        return model, encoder_model, decoder_model
    
    #評価
    def eval_perplexity(self,model,e_test,d_test,t_test,batch_size) :
        row = test.shape[0]
        list_loss = []

        s_time = time.time()
        n_batch = math.ceil(row/batch_size)

        n_loss = 0
        sum_loss = 0.
        for i in range(0,n_batch) :
            s = i * batch_size
            e = min([(i+1)*batch_size,row])
            e_on_batch = e_test[s:e,:]
            d_on_batch = d_test[s:e,:]
            t_on_batch = t_test[s:e,:]
            t_on_batch = np_utils.to_categorical(t_onbatch,self.output_dim)
            #mask行列作成
            mask1 = np.zeros((e-s,self.maxlen_d,self.output_dim),dtype=np.float32)
            for j in range(0,e-s) :
                n_dim = maxlen_d - list(d_on_batch[j,:]).count(0.)
                mask1[j,0:n_dim,:] = 1
                n_loss += n_dim

            mask2 = mask1.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            #予測
            y_predict1 = model.predict_on_batch([e_on_batch,d_on_batch])
            y_predict2 = np.maxium(y_predict1,1e-7)
            y_predict2 = -np.log(y_predict2)
            y_predict3 = y_predict2.reshape(1,(e-s)*self.maxlen_d*self.output_dim)

            target = t_on_batch.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            #マスキング
            target1 = target * mask2
            #category_cross_entropy計算
            loss = np.dot(y_predict3,target.T)
            sum_loss += loss[0,0]
            #perplexity計算
            perplexity = pow(math.e,sum_loss/n_loss)
            elapsed_time = time.time() - s_time
            sys.stdout.write("\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s "+"\t"+
                                "{0:.4f}".format(perplexity)+"                 ")
            sys.stdout.flush()
            del e_on_batch, d_on_batch, t_on_batch
            del mask1, mask2
            del y_predict1, y_predict2, y_predict3
            del target
            gc.collect()

        print()

        return perplexity

    #train_on_batchメイン処理
    def on_batch(self,model,j,e_train,d_train,t_train,e_val,d_val,t_val,batch_size) :
        #損失関数、評価関数の平均計算用リスト
        list_loss = []
        list_accuracy = []

        s_time = time.time()
        row = e_train.shape[0]
        n_batch = math.ceil(row/batch_size)
        for i in range(0,n_batch) :
            s = i*batch_size
            e = min([(i+1)*batch_size,row])
            e_on_batch = e_test[s:e,:]
            d_on_batch = d_test[s:e,:]
            t_on_batch = t_test[s:e,:]
            t_on_batch = np_utils.to_categorical(t_on_batch,self.output_dim)
            result = model.train_on_batch([e_on_batch,d_on_batch],t_on_batch)
            list_loss.append(result[0])
            list_accuracy.append(result[1])
            #perplexity = pow(math.e,np.average(list_loss))
            elapsed_time = time.time() - s_time
            sys.stdout.write("\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s "+"\t"+
                            "{0:.4f}".format(np.average(list_loss))+"\t"+
                            "{0:.4f}".format(np.average(list_accuracy)))
            sys.stddout.flush
            del e_on_batch, d_on_batch, t_on_batch
        
        #perplexity評価
        print()
        val_perplexity = self.eval_perplexity(model,e_val,d_val,t_val,batch_size)

        del list_loss, list_accuracy

        return val_perplexity

    #学習
    def train(self,e_input,d_input,target,batch_size,epochs,emb_param) :
        print("#1",target.shape)
        model, _, _ = self.create_model
        if os.path.isfile(embparam) :
            #埋め込みパラメータセット
            model.load_weigths(emb_param)
        
        print("#6")
        e_i = e_input
        d_i = d_input
        t_l = target
        z = list(zip(e_i,d_i,t_l))
        nr.shuffle(z)
        e_i, d_i, t_l = zip(*z)

        e_i = np.array(e_i).reshape(len(e_i),self.maxlen_e)
        d_i = np.array(d_i).reshape(len(d_i),self.maxlen_d)
        t_l = np.array(t_l).reshape(len(t_l),self.maxlen_d)

        #訓練データとテストデータを9:1に分割
        n_split = int(e_i.shape[0]*0.9)
        e_train, e_val = np.vsplit(e_i,[n_split])
        d_train, d_val = np.vsplit(d_i,[n_split])
        t_train, t_val = np.vsplit(t_l,[n_split])

        row = e_input.shape[0]
        loss_bk = 10000
        for j in range(0,epochs) :
            print("Epoch ",j+1,'/',epochs)
            val_perplexity = self.on_batch(model,j,e_train,d_train,t_train,e_val,d_val,t_val,batch_size)

            #Early_Stopping
            if j == 0 or val_perplexity <= loss_bk :
                loss_bk = val_perplexity
            else :
                print("Early_Stopping")
                break

        return model

    #応答文生成
    def response(self,e_input,length) :
        encoder_outputs, state_h, state_c = encoder_model.predict(e_input)
        states_value = [state_h,state_c]

        #空のターゲット生成
        target_seq = np.zeros((1,1))
        target_seq[0,0] = words_indices["SSSS"]

        decoded_sentence = ''
        for i in range(0,length) :
            output_tokens, h, c = decoder_model.predict([target_seq]+states_value)

            sampled_token_index = np.argmax(output_tokens[0,0,:])
            sampled_char = indices_words[sampled_token_index]

            if sampled_char == "SSSS" :
                break
            decoded_sentence += sampled_char

            if i == length-1 :
                break
            target_seq[0,0] = sampled_token_index
            states_value = [h,c]

        return decoded_sentence

#実行処理

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
    t = pickle

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
batch_