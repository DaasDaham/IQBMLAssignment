from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import gensim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
df=pd.read_csv(r'train.csv')
df2=pd.read_csv(r'test.csv')
df3=pd.read_csv(r'sample.csv')
df2=pd.merge(df2,df3,left_on='ID',right_on="ID")
df=df.rename(columns={"Lable":"Label"})
df=pd.concat([df,df2])
X=df[["Sequence"]]
y=df[["Label"]]
#print(df)
codes = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10,'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}
encode_list = []
for row in df['Sequence'].values:
    row_encode = []
    for code in row:
      row_encode.append(codes.get(code, 0))
    encode_list.append(np.array(row_encode))

#print(encode_list)
#from keras.preprocessing.sequence import pad_sequences
#max_length = 100
#train_pad = pad_sequences(encode_list, maxlen=max_length, padding='post', truncating='post')
#print(train_pad.shape)
import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)
objects = []
with (open("data4.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
print(objects)
#print(encode_list)
#X_train, X_test, y_train, y_test = train_test_split(X, y,
#random_state=42)
#forest = RandomForestClassifier(n_estimators=5, random_state=2)
#forest.fit(X_train, y_train)