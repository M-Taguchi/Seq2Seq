# -*- coding: utf-8 -*-

import numpy as np
import csv
import glob
import re

def make_data(fname, data2):
    f = open(fname, 'r', encoding="utf-8")
    df1 = csv.reader(f)

    data1 = [v for v in df1]

    print(len(data1))
    # file 読み込み
    text = ''
    for i in range(0,len(data1)):
        if len(data1[i]) == 0:
            print("null")
            continue

        s = data1[i][0]
        if s[0:5] == "％ｃｏｍ：" :
            continue
        if s[0] != '＠' :
            #不明文字をUNKに置き換え
            s = s.replace("＊＊＊","UNK")
            #会話文セパレータ
            if s[0] == 'F' or s[0] == 'M' :
                s = "SSSS" + s[5:]
            if s[0:2] == "Ｘ：" :
                s = "SSSS"+s[2:]
            
            s = re.sub("F[0-9]{3}","UNK",s)
            s = re.sub("M[0-9]{3}","UNK",s)
            s = s.replace('＊',"")
        else :
            continue

        #補足説明除去
        while s.find('（') != -1 :
            start_1 = s.find('（')
            if s.find('）') != -1 :
                end_1 = s.find('）')
                if start_1 >= end_1 :
                    s = s.replace(s[end_1],"")
                else :
                    s = s.replace(s[start_1:end_1+1],"")
                if len(s) == 0 :
                    continue
            else :
                s = s[0:start_1]

        while s.find('［') != -1 :
            start_2 = s.find('［')
            if s.find('］') != -1 :
                end_2 = s.find('］')
                if start_2 >= end_2 :
                    s = s.replace(s[end_2],"")
                else :
                    s = s.replace(s[start_2:end_2+1],"")
                if len(s) == 0 :
                    continue
            else :
                s = s[0:start_2]

        while s.find('＜') != -1 :
            start_3 = s.find('＞')
            if s.find('＜') != -1 :
                end_3 = s.find('＞')
                if start_3 >= end_3 :
                    s = s.replace(s[end_3],"")
                else :
                    s = s.replace(s[start_3:end_3+1],"")
                if len(s) == 0 :
                    continue
            else :
                s = s[0:start_3]

        while s.find('【') != -1 :
            start_4 = s.find('【')
            if s.find('】') != -1 :
                end_4 = s.find('】')
                if start_4 >= end_4 :
                    s = s.replace(s[end_4],"")
                else :
                    s = s.replace(s[start_4:end_4+1],"")
                if len(s) == 0 :
                    continue
            else :
                s = s[0:start_4]

        #削除後の残った文字を出力文字列に追加
        if s != "\n" and s != "SSSS" :
            text += s

    #セパレータごとにファイル書き込み
    text = text[4:]
    while text.find("SSSS") != -1 :
        end_s = text.find("SSSS")
        t = text[0:end_s]
        #長い会話文を分割
        if end_s > 100 :
            while len(t) > 100 :
                if t.find('。') != -1 :
                    n_period = t.find('。')
                    data2.append("SSSS"+t[0:n_period+1])
                    t = t[n_period+1:]
                else :
                    break
        data2.append("SSSS"+t)
        text = text[end_s+4:]
    f.close()
    return

file_list = glob.glob("nucc/*")
print(len(file_list))

data2 = []
for j in range(0,len(file_list)) :
    print(file_list[j])
    make_data(file_list[j],data2)

#file save
f = open("nucc2/corpus.txt", 'w', encoding="utf-8")
for i in range(0,len(data2)) :
    f.write(str(data2[i])+'\n')
f.close()
print(len(data2))