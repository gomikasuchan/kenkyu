import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop

df = pd.read_csv('D:/20180615_立石.csv')
L = len(df)
tag = df.iloc[:, [3,4]]         #発話意図タグの列のみ抽出py
tag = tag.dropna()          #欠損値を削除
tag_com = tag["タグ"].str.cat(tag['サブタグ１'])
tag_list = tag_com.values       #listに変換 

# textから重複しない文字のセットを抽出してlistに変換してソート
keys = sorted(list(set(tag_list)))

# 重複しない文字と文字の数を表示
#print(keys)
#print(len(keys))

#サブタグ込みのタグを一文字にするために辞書を作成
chars = {'a','b','c','d','e','f','g','h','i','j','k','l'}
tag_dic = dict(zip(keys, chars))
key_dic = dict(zip(chars, keys))

#作成した辞書を標示
#print(tag_dic)
#print(key_dic)

#作成した辞書を用いてリスト内のサブタグ込みのタグを一文字に変換
tag_new_list = []
for yoso in tag_list :
    if yoso in tag_dic:
        val = tag_dic[yoso]
        tag_new_list.append(val)

#変換後のリストを標示
print(tag_new_list)
print(len(tag_new_list))

#リストを文字列にする
tag_text = ''.join(tag_new_list)

#作成した文字列を標示
print(tag_text)

maxlen = 6
step = 1
sentences = []  # 入力データ
next_chars = [] # 正解データ
for i in range(0, len(tag_text) - maxlen, step):
    sentences.append(tag_text[i: i + maxlen])
    next_chars.append(tag_text[i + maxlen])

# sentencesのの内容を確認
print(len(sentences))
print(next_chars)

maxlen = 6
step = 1
sentences = []  # 入力データ
next_chars = [] # 正解データ
for i in range(0, len(tag_text) - maxlen, step):
    sentences.append(tag_text[i: i + maxlen])
    next_chars.append(tag_text[i + maxlen])

# sentencesのの内容を確認
print(len(sentences))
print(next_chars)

# textから重複しない文字のセットを抽出してlistに変換してソート
chars = sorted(list(set(tag_text)))
# 重複しない文字の数を表示
print(len(chars))
# 文字に番号をふり、文字から番号、番号から文字の辞書を作成
char_indices = dict((char,i) for i, char in enumerate(chars))
indices_char = dict((i,char) for i, char in enumerate(chars))
# 文字から番号の辞書の内容を確認
print(char_indices)

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) # 入力データ
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # 正解データ
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print(X)
print(y)

X_Train = X[0:368]
y_Train = y[0:368]
X_Test = X[369:]
y_Test = y[369:]

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 学習
result = model.fit(X_Train, y_Train, batch_size=25, epochs=30, verbose=1, validation_data=(X_Test, y_Test))

# テストデータに対して正解率を表示
#model.compile()で指定した損失関数(loss)，評価関数(matrics)の結果がかえってくる
score = model.evaluate(X_Test, y_Test, verbose=1)
score1 = model.evaluate(X_Train, y_Train, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Train score:', score1[0])
print('Train accuracy:', score1[1])

# 【モデルを保存】
model.save("D:/tetsuko_lstm.h5")
# 【モデルの重みを保存】
model.save_weights("D:/tetsuko_lstm_weight.h5")