# -*- coding: utf-8 -*-

import numpy as np
import pickle

#1次元配列ロード
mat_corpus = np.load("mat_corpus.npy")
data1 = [v[0] for v in mat_corpus]

mat1 = np.array(data1).reshape((len(data1),1))

mat0 = ["SSSS"]
mat0 = np.array(mat0).reshape((1,1))

mat = np.r_[mat0[:,0],mat1[:,0]]

words = sorted(list(set(mat)))
cnt = np.zeros(len(words))

print("total words:",len(words))
#単語をキーにインデックス検索
words_indices = dict((w,i) for i,w in enumerate(words))
#インデックスをキーに単語検索
indices_words = dict((i,w) for i,w in enumerate(words))

#単語の出現回数をカウント
for j in range(0,len(mat)) :
    cnt[words_indices[mat[j]]] += 1

#出現頻度の少ない単語を「UNK」で置き換え
words_unk = []

for k in range(0,len(words)) :
    if cnt[k] <= 3 :
        words_unk.append(words[k])
        words[k] = "UNK"

print("words_unk:",len(words_unk))

#低頻度単語をUNKに置き換えたので、辞書作り直し
words = list(set(words))
#0padding対策
words.append('\t')
words = sorted(words)
print("new total words:",len(words))

words_indices = dict((w,i) for i,w in enumerate(words))
#インデックスをキーに単語検索
indices_words = dict((i,w) for i,w in enumerate(words))

#単語インデックス配列作成
mat_urtext = np.zeros((len(mat),1),dtype=int)
for i in range(0,len(mat)) :
    if mat[i] in words_indices :
        mat_urtext[i,0] = words_indices[mat[i]]
    else :
        mat_urtext[i,0] = words_indices["UNK"]

print(mat_urtext.shape)

#作成した辞書をセーブ
with open('words_indices.pickle', 'wb') as f :
    pickle.dump(words_indices , f)

with open('indices_words.pickle', 'wb') as g :
    pickle.dump(indices_words , g)

#単語ファイルセーブ
with open('words.pickle', 'wb') as h :
    pickle.dump(words , h)

#コーパスセーブ
with open('mat_urtext.pickle', 'wb') as ff :
    pickle.dump(mat_urtext , ff) 