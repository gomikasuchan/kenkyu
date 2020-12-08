
import warnings
import pandas as pd
import numpy as np
import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import models

warnings.simplefilter('ignore')

#データの読み込み
print("filename please")
csv_name = input()
df = pd.read_csv(csv_name)
sentence = df.iloc[:, 2]            #発話の列のみ抽出
sentence = sentence.fillna('n')        #欠損値を削除
sentence_list = sentence.values     #listに変換

tag = df.iloc[:, 3]                 #発話意図タグの列のみを抽出
tag = tag.fillna('n')                #欠損地を穴埋め
tag_list = tag.values               #listに変換



"""
#空白で単語を区切り、改行で文書を区切っているテキストファイルを作成する
tagger = MeCab.Tagger("-Owakati")
with open('tetsuko_sentence.txt', 'a') as f:
    for sentence in sentence_list:
        str_output = tagger.parse(sentence) #str型で、単語が空白で別れる
        f.write("%s\n" % str_output)
"""
sentence_list2 = []

for sentence in sentence_list:
    tagger = MeCab.Tagger("-Owakati")
    str_output = tagger.parse(sentence)
    sentence_list2.append(str_output)

sentence_list3 = []
for sentence in sentence_list2:
    sentence_list3.append(sentence.split())

docs = sentence_list3
tags = tag_list 

print(docs)
print(tags)

class LabeledListSentence(object):
    def __init__(self, word_list, labels):
        self.word_list = word_list
        self.labels = labels
    
    def __iter__(self):
        for i, words in enumerate(self.word_list):
            yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])

#gnesimに登録
#文章にタグを付与する
sentences = LabeledListSentence(docs,tags)

#学習条件
#alpha:　学習率 / min_count: x回未満しか出てこない単語は無視（ここでは使わない）
#size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数 
#epochs : 1.データセットをバッチサイズに従ってN個のサブセットに分ける
#         2.各サブセットを学習に回す。つまり、N回学習を繰り返す(iter)
#         1,2を何回実行するかを決めるのがepochs
#         word2vecの場合、30が安定するらしい

model = Doc2Vec(alpha=0.025, size=100, iter=20, workers=4, epochs=30)

#doc2vecの学習前準備（単語リスト構築）
model.build_vocab(sentences)

#学習実行
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

#セーブ
model.save('d2v_tetsuko.model')

#順番が変わることがあるのでタグリストは学習後に再呼び出し
tags = model.docvecs.offset2doctag


