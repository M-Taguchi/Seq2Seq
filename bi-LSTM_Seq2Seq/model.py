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
import keras

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

class Seq2SeqModel :
    def __init__(self,maxlen_e,maxlen_d,n_hidden,input_dim,vec_dim,output_dim) :
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
        
        #前向き1段目
        e_i_fw1, state_h_fw1, state_c_fw1 = LSTM(self.n_hidden,name="encoder_LSTM_fw1",
                                                return_sequences=True,return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20190415),
                                                recurrent_initializer=orthogonal(gain=1.0,seed=20190415))(e_i)
        
        #前向き2段目
        e_i_fw2, state_h_fw2, state_c_fw2 = LSTM(self.n_hidden,name="encoder_LSTM_fw2",
                                                return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20190415),
                                                recurrent_initializer=orthogonal(gain=1.0,seed=20190415),dropout=0.5,recurrent_dropout=0.5)(e_i_fw1)
        e_i_bw0 = e_i
        #後ろ向き1段目
        e_i_bw1, state_h_bw1, state_c_bw1 = LSTM(self.n_hidden,name="encoder_LSTM_bw1",
                                                return_sequences=True,return_state=True,go_backwards=True,
                                                kernel_initializer=glorot_uniform(seed=20190415),
                                                recurrent_initializer=orthogonal(gain=1.0,seed=20190415))(e_i_bw0)
        
        #後ろ向き2段目
        e_i_bw2, state_h_bw2, state_c_bw2 = LSTM(self.n_hidden,name="encoder_LSTM_bw2",
                                                return_state=True,go_backwards=True,
                                                kernel_initializer=glorot_uniform(seed=20190415),
                                                recurrent_initializer=orthogonal(gain=1.0,seed=20190415),dropout=0.5,recurrent_dropout=0.5)(e_i_bw1)

        """
        encoder_outputs, state_h, state_c = LSTM(self.n_hidden,name="encoder_LSTM",
                                                return_state=True,
                                                kernel_initializer=glorot_uniform(seed=20190415),
                                                recurrent_initializer=orthogonal(gain=1.0,seed=20190415),dropout=0.5,recurrent_dropout=0.5)(e_i)
        """

        encoder_outputs = keras.layers.add([e_i_fw2,e_i_bw2],name="encoder_outputs")

        #'encoder_outputs'は使わない。statesのみ使用する
        state_h_1 = keras.layers.add([state_h_fw1,state_h_bw1])
        state_c_1 = keras.layers.add([state_c_fw1,state_c_bw1])
        state_h_2 = keras.layers.add([state_h_fw2,state_h_bw2])
        state_c_2 = keras.layers.add([state_c_fw2,state_c_bw2])
        encoder_states1 = [state_h_1,state_c_1]
        encoder_states2 = [state_h_2,state_c_2]
        #encoder_states = [state_h, state_c]

        encoder_model = Model(inputs=encoder_input,
                             outputs=[encoder_outputs,state_h_1,state_c_1,state_h_2,state_c_2])
        #encoder_model = Model(inputs=encoder_input,outputs=[encoder_outputs,state_h,state_c])
        
        print("#3")
        #デコーダ(学習用)
        #レイヤ定義
        decoder_LSTM1 = LSTM(self.n_hidden,name="decoder_LSTM1",
                            return_sequences=True,return_state=True,
                            kernel_initializer=glorot_uniform(seed=20190415),
                            recurrent_initializer=orthogonal(gain=1.0,seed=20190415))
        
        decoder_LSTM2 = LSTM(self.n_hidden,name="decoder_LSTM2",
                            return_sequences=True,return_state=True,
                            kernel_initializer=glorot_uniform(seed=20190415),
                            recurrent_initializer=orthogonal(gain=1.0,seed=20190415),
                            dropout=0.5,recurrent_dropout=0.5)

        """
        decoder_LSTM = LSTM(self.n_hidden,name="decoder_LSTM",
                           return_sequences=True,return_state=True,
                           kernel_initializer=glorot_uniform(seed=20190415),
                           recurrent_initializer=orthogonal(gain=1.0,seed=20190415),dropout=0.5,recurrent_dropout=0.5)
        """
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
        d_i_1, h1, c1 = decoder_LSTM1(d_i,initial_state=encoder_states1)
        d_i_2, h2, c2 = decoder_LSTM2(d_i_1,initial_state=encoder_states2)
        #d_outputs, _, _ = decoder_LSTM(d_i,initial_state=encoder_states)
        print("#4")
        decoder_outputs = decoder_Dense(d_i_2)
        #decoder_outputs = decoder_Dense(d_outputs)
        model = Model(inputs=[encoder_input,decoder_inputs],outputs=decoder_outputs)
        model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["categorical_accuracy"])

        #デコーダ(応答文生成)
        print("#5")
        #入力定義
        decoder_state_input_h_1 = Input(shape=(self.n_hidden,),name="input_h_1")
        decoder_state_input_c_1 = Input(shape=(self.n_hidden,),name="input_c_1")
        decoder_states_inputs_1 = [decoder_state_input_h_1,decoder_state_input_c_1]
        decoder_state_input_h_2 = Input(shape=(self.n_hidden,),name="input_h_2")
        decoder_state_input_c_2 = Input(shape=(self.n_hidden,),name="input_c_2")
        decoder_states_inputs_2 = [decoder_state_input_h_2,decoder_state_input_c_2]
        decoder_states_inputs = [decoder_state_input_h_1,decoder_state_input_c_1,
                                 decoder_state_input_h_2,decoder_state_input_c_2]

        """
        decoder_state_input_h = Input(shape=(self.n_hidden,),name="input_h")
        decoder_state_input_c = Input(shape=(self.n_hidden,),name="input_c")
        decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]
        """
        #LSTM
        d_i_1_2, state_h_1, state_c_1 = decoder_LSTM1(d_input,initial_state=decoder_states_inputs_1)
        d_i_2_2, state_h_2, state_c_2 = decoder_LSTM2(d_i_1_2,initial_state=decoder_states_inputs_2)
        decoder_states = [state_h_1,state_c_1,state_h_2,state_c_2]
        decoder_res = decoder_Dense(d_i_2_2)
        decoder_model = Model

        """
        decoder_lstm, state_h, state_c = decoder_LSTM(d_input,initial_state=decoder_states_inputs)
        decoder_states = [state_h,state_c]
        decorder_res = decoder_Dense(decoder_lstm)
        """

        decoder_model = Model([decoder_inputs]+decoder_states_inputs,[decoder_res]+decoder_states)

        return model, encoder_model, decoder_model
    
    #評価
    def eval_perplexity(self,model,e_test,d_test,t_test,batch_size) :
        row = e_test.shape[0]
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
            t_on_batch = np_utils.to_categorical(t_on_batch,self.output_dim)
            #mask行列作成
            mask1 = np.zeros((e-s,self.maxlen_d,self.output_dim),dtype=np.float32)
            for j in range(0,e-s) :
                n_dim = self.maxlen_d - list(d_on_batch[j,:]).count(0.)
                mask1[j,0:n_dim,:] = 1
                n_loss += n_dim

            mask2 = mask1.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            #予測
            y_predict1 = model.predict_on_batch([e_on_batch,d_on_batch])
            y_predict2 = np.maximum(y_predict1,1e-7)
            y_predict2 = -np.log(y_predict2)
            y_predict3 = y_predict2.reshape(1,(e-s)*self.maxlen_d*self.output_dim)

            target = t_on_batch.reshape(1,(e-s)*self.maxlen_d*self.output_dim)
            #マスキング
            target1 = target * mask2
            #category_cross_entropy計算
            loss = np.dot(y_predict3,target1.T)
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
            s = i * batch_size
            e = min([(i+1)*batch_size,row])
            if e != (i+1)*batch_size :
                print(e)
            e_on_batch = e_train[s:e,:]
            e_on_batch = np.reshape(e_on_batch,(len(e_on_batch),self.maxlen_e))
            d_on_batch = d_train[s:e,:]
            d_on_batch = np.reshape(d_on_batch,(len(d_on_batch),self.maxlen_d))
            t_on_batch = t_train[s:e,:]
            t_on_batch = np.reshape(t_on_batch,(len(t_on_batch),self.maxlen_d))
            t_on_batch = np_utils.to_categorical(t_on_batch,self.output_dim)
            result = model.train_on_batch([e_on_batch,d_on_batch],t_on_batch)
            list_loss.append(result[0])
            list_accuracy.append(result[1])
            """
            if np.isnan(result[0]) :
                print()
                print(s, e, i, e_on_batch)
            """
            #perplexity = pow(math.e,np.average(list_loss))
            elapsed_time = time.time() - s_time
            sys.stdout.write("\r"+str(e)+"/"+str(row)+" "+str(int(elapsed_time))+"s "+"\t"+
                            "{0:.4f}".format(np.average(list_loss))+"\t"+
                            "{0:.4f}".format(np.average(list_accuracy)))
            sys.stdout.flush
            del e_on_batch, d_on_batch, t_on_batch
        
        #perplexity評価
        print()
        val_perplexity = self.eval_perplexity(model,e_val,d_val,t_val,batch_size)

        del list_loss, list_accuracy

        return val_perplexity

    #学習
    def train(self,e_input,d_input,target,batch_size,epochs,emb_param) :
        print("#1",target.shape)
        model, _, _ = self.create_model()
        if os.path.isfile(emb_param) :
            #埋め込みパラメータセット
            model.load_weights(emb_param)
        
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
    def response(self,enc_model,dec_model,e_input,length,words_indices,indices_words) :
        encoder_outputs, state_h_1, state_c_1, state_h_2, state_c_2 = enc_model.predict(e_input)
        states_value = [state_h_1,state_c_1,state_h_2,state_c_2]

        #空のターゲット生成
        target_seq = np.zeros((1,1))
        target_seq[0,0] = words_indices["SSSS"]

        decoded_sentence = ''
        for i in range(0,length) :
            output_tokens, h1, c1, h2, c2 = dec_model.predict([target_seq]+states_value)

            sampled_token_index = np.argmax(output_tokens[0,0,:])
            sampled_char = indices_words[sampled_token_index]

            if sampled_char == "SSSS" :
                break
            decoded_sentence += sampled_char

            if i == length-1 :
                break
            target_seq[0,0] = sampled_token_index
            states_value = [h1,c1,h2,c2]

        return decoded_sentence
