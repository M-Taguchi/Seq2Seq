# -*- coding: utf-8 -*-

import numpy as np
import csv

def generate_npy(source_csv,generated_npy) :
    df2 = csv.reader(open(source_csv,'r',encoding='utf-8'),delimiter=' ')

    data2 = [v for v in df2]

    mat = np.array(data2)
    print(mat.shape)
    mat_corpus = []

    #補正
    for i in range(0,mat.shape[0]) :
        if mat[i][0] != '@' and mat[i][0] != "EOS" and mat[i][0] != '┐' and mat[i][0] != '┘' :
            if mat[i][0] == "SSSSUNK：" :
                mat_corpus.append("SSSS")
            elif mat[i][0] == "SSSSUNKUNK" :
                mat_corpus.append("SSSS")
            elif len(mat[i][0]) > 4 and mat[i][0][0:4] == "SSSS" :
                mat_corpus.append("SSSS")
                mat_corpus.append(mat[i][0][4:])
            elif mat[i][0] == "UNK：" or mat[i][0] == "X：" :
                mat_corpus.append("SSSS")
            elif mat[i][0] == "UNKUNK" :
                mat_corpus.append("UNK")
            else :
                mat_corpus.append(mat[i][0])

    #デリミタ連続対策

    mat_corpus1 = []

    for i in range(1,len(mat_corpus)) :
        if mat_corpus[i] == "SSSS" and mat_corpus[i-1] == "SSSS" :
            continue
        else :
            mat_corpus1.append((mat_corpus[i]))

    mat_corpus1.append("SSSS")
    mat_corpus1 = np.array(mat_corpus1).reshape(len(mat_corpus1),1)
    #コーパス行列セーブ
    np.save(generated_npy,mat_corpus1)
    print(mat_corpus1.shape)

    return

generate_npy('nucc2/result.csv','mat_corpus.npy')