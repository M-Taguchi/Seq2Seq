# -*- coding: utf-8 -*-
import numpy.random as nr
import numpy as np
import pickle

#辞書のロード
with open('words_indices.pickle', 'rb') as f :
    words_indices = pickle.load(f)         

with open('indices_words.pickle', 'rb') as g :
    indices_words = pickle.load(g)

#単語ファイルのロード
with open('words.pickle', 'rb') as h :
    words = pickle.load(h)

#コーパスのロード
with open('mat_urtext.pickle', 'rb') as ff :
    mat_urtext = pickle.load(ff)

#入力語数
max_len_e = 50
#出力語数
max_len_d = 50

#コーパスを会話文のリストに変換
separater = words_indices["SSSS"]
data = []

for i in range(0,mat_urtext.shape[0]-1) :
    if mat_urtext[i,0] == separater :
        dialog = []
    else :
        dialog.append(mat_urtext[i,0])
    if mat_urtext[i+1,0] == separater :
        data.append(dialog)

print(len(data))

#encode_input_data
enc_input = data[:-1]

#decode_input_data
dec_input = []
for i in range(1,len(data)) :
    enc_dialog = data[i][:]
    enc_dialog.insert(0,separater)
    dec_input.append(enc_dialog)

#target
target = []
for i in range(1,len(data)) :
    dec_dialog = data[i][:]
    dec_dialog.append(separater)
    target.append(dec_dialog)

e_input = []
d_input = []
t_l = []
for i in range(len(enc_input)) :
    if len(enc_input[i]) <= max_len_e and len(dec_input[i]) <= max_len_d :
        e_input.append(enc_input[i][:])
        d_input.append(dec_input[i][:])
        t_l.append(target[i][:])

#0padding

for i in range (0,len(e_input)) :
    #リストの後ろに0追加
    e_input[i].extend([0]*max_len_e)
    d_input[i].extend([0]*max_len_d)
    t_l[i].extend([0]*max_len_d)
    #系列長で切り取り
    e_input[i] = e_input[i][0:max_len_e]
    d_input[i] = d_input[i][0:max_len_d]
    t_l[i] = t_l[i][0:max_len_d]

#リストから配列に変換
e = np.array(e_input).reshape(len(e_input),max_len_e,1)
d = np.array(d_input).reshape(len(d_input),max_len_d,1)
t = np.array(t_l).reshape(len(t_l),max_len_d,1)

#シャッフル
z = list(zip(e, d, t))
nr.seed(12345)
e,d,t = zip(*z)
nr.seed()

e = np.array(e).reshape(len(e_input),max_len_e,1)
d = np.array(d).reshape(len(d_input),max_len_d,1)
t = np.array(t).reshape(len(t_l),max_len_d,1)

print(e.shape,d.shape,t.shape)
print(e[1])

#Encoder Inputデータをセーブ
with open('e.pickle', 'wb') as f :
    pickle.dump(e , f)

#Decoder Inputデータをセーブ
with open('d.pickle', 'wb') as g :
    pickle.dump(d , g)

#ラベルデータをセーブ
with open('t.pickle', 'wb') as h :
    pickle.dump(t , h)

#maxlenセーブ
with open('maxlen.pickle', 'wb') as maxlen :
    pickle.dump([max_len_e, max_len_d] , maxlen)