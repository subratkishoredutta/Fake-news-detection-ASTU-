# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:13:24 2020

@author: Asus
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

stop_words= set(stopwords.words("english"))

lemma =  WordNetLemmatizer()

news = pd.read_csv(r"news.csv")

print(news['title'].head())

data=news.drop(['Unnamed: 0'],axis=1)

TEXTdata=[]
TITLEdata=[]
##data cleaning and formating
for i in range(len(news)):
    data['text'].iloc[i] = re.sub('[^a-zA-Z]',' ',data['text'].iloc[i]).lower()
    data['title'].iloc[i] = re.sub('[^a-zA-Z]',' ',data['title'].iloc[i]).lower()
    
    textword = word_tokenize(data['text'].iloc[i])
    titleword = word_tokenize(data['title'].iloc[i])
    text=""
    title=""
    for w in textword:
        if w not in stop_words:
            wr = lemma.lemmatize(w)
            text=text+" "+wr
    for k in titleword:
        if k not in stop_words:
            kr = lemma.lemmatize(k)
            title=title+" "+kr
    TEXTdata.append(text)
    TITLEdata.append(title)       

## label creation
    
Y=[]

for i in range(len(data)):
    if data['label'].iloc[i] == 'FAKE':
        Y.append(1)
    elif data['label'].iloc[i] == 'REAL':
        Y.append(0)

##text to vector

cv=TfidfVectorizer()
X=cv.fit_transform(TEXTdata).toarray()



##split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2)
##logs


##model
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1024,input_shape=(X.shape[1],),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(512,input_shape=(1024,),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1024,input_shape=(512,),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1,input_shape=(1024,),activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(Xtrain,Ytrain,batch_size=64 ,epochs=100)
##training curves
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()


Ypred=model.predict(Xtest)
Ypred=(Ypred>0.5).astype(np.uint8)

CF=confusion_matrix(Ytest,Ypred)
print("CONFUSION MATRIX:\n",CF)
CLFR= classification_report(Ytest,Ypred)

print ("CLASSIFICATION REPORT:\n",CLFR)

