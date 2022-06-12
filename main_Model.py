#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
from os import path 
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import emoji
import string
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Embedding, Flatten, Dropout, Bidirectional, LSTM, Lambda
from tensorflow.keras import layers,Model
from keras.models import Sequential
import keras.backend as K

from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertTokenizerFast,BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[8]:


class detection_model(tf.keras.Model):
    def __init__(self, xtrain,xval,ytrain,yval,n_in,
                 classfier_method = 'FC_NN',
                 weight_path = False):
        super(detection_model, self).__init__()
        self.xtrain = xtrain
        self.xval = xval
        self.ytrain = ytrain
        self.yval = yval
        self.n_in = n_in
        
        if classfier_method == 'FC_NN':
            self.classifer = FC_NN(self.n_in)
        else:
            self.classifer = Lstm(self.n_in)
        
        if weight_path != False:
            self.classifer.load_weights(weight_path)
        
    def train_model(self,los,opt,n_epoch,n_batch, 
                         save_weight_path = False):
        
        self.classifer.compile(loss = los,optimizer = opt,metrics = ['accuracy'])
        
        print("Training full_model: ")
        self.classifer.fit(self.xtrain, self.ytrain, validation_data=(self.xval,self.yval), 
                           epochs=n_epoch, batch_size=n_batch, verbose=1)
        print("Done!!!!")
        
        if save_weight_path != False:
            self.classifer.save_weights(save_weight_path)
            
        return None
    
    def pre_use(self,k):
        a = self.classifer.predict(k)
        b = sum(np.argmax(a,axis = 1)) / 100

        return round(b)
    
    def evaluate_model(self,model,xtest,ytest,val_batch,OUT='per tweet'):
        
        if OUT == 'per tweet':
            print("Evaluate on test data per tweet")
            results = self.classifer.evaluate(xtest, ytest, batch_size = val_batch)
            print("test loss, test acc:", results)
        
        else:
            print("Evaluate on test data per user")
            y_pred = []
            for i in range(60):
                y_pred.append(predict_user(xtest))
            
            results = accuracy_score(ytest, np.array(y_pred), normalize=True)
            
            print("test loss, test acc:", results)
        
        
    def predict_tweet(self,sent,no_bert = True):
        sent_feature = get_Feature(sent)
        if bert:
            sent_feature = sent_feature[:11]
        
        return self.classifer.predict(sent_feature)
        
    def predict_user(self,data):
        n_data = len(data)
        data_feature = []
        for i in range(n_data):
            data_feature.append(get_Feature(data[i]))
            
        data_feature = np.array(data_feature).reshape((n_data,-1))
    
        pred = f_model.predict(data)
        Class = sum(np.argmax(pred,axis = 1)) / 100

        return round(Class)        


# In[6]:


class FC_NN(tf.keras.Model):
    def __init__(self, n_input):
        super(FC_NN, self).__init__()
        
        input_feature = layers.Input(shape=(n_input),dtype='float32')
        x = layers.Dense(64, activation='relu',name = 'impo')(input_feature)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(rate=0.3)(x)


        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(rate=0.3)(x)


        output = layers.Dense(2, activation='softmax')(x)
        self.f_model = Model(inputs=input_feature, outputs=output)
        
    def build_model(self):
        return self.f_model


# In[5]:


class Lstm(tf.keras.Model):
    def __init__(self, embedding_model):
        super(Lstm, self).__init__()
        
        input_feature = layers.Input(shape=(n_input),dtype='float32')
        x = layers.Dense(64, activation='relu',name = 'impo')(input_feature)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(rate=0.3)(x)

        x = Lambda(lambda x:K.expand_dims(x,axis = 1))(x)

        x = layers.LSTM(32,activation='tanh',return_sequences = True)(x)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.LSTM(16,activation='tanh',return_sequences = True)(x)
        x = layers.Dropout(rate=0.3)(x)


        output = layers.Dense(2, activation='softmax')(x)
        self.lstm_model = Model(inputs=input_feature, outputs=output)
        
    def build_model(self):
        return self.lstm_model


# In[4]:


class Sort_data():
    def __init__(self,data_path,label_path,n_user,n_vec):
        self.label_path = label_path
        self.data_path = data_path
        self.n_user = int(n_user)
        self.n_vec = int(n_vec)
        
    def get_label(self):
        label = pd.read_csv(self.label_path, sep=":::", header=None)
        for i in range(420):
            if label.iloc[i,1] == 'I':
                label.iloc[i,1] = 1
            else:
                label.iloc[i,1] = 0
        label.columns = ['name','label']
        
        return label
    
    def read_xml2(self,url):
        tree = ET.parse(url)
        root = tree.getroot()
        b_document = []
        for documents in root:
            for document in documents:
                b_document.append(document.text)
                
        return b_document
    
    def get_sentence(self):
        path = self.data_path
        path_list = os.listdir(path)
        path_list.sort()
        k = 0
        xml_name = []
        for filename in path_list:
            xml_name.append(os.path.join(path,filename))
            k += 1
            
        all_sentence =  []
        for xml in xml_name:
            all_sentence = all_sentence + self.read_xml2(xml)
            
        return all_sentence
    
    def merge_sent_label(self):
        all_sentence = pd.DataFrame(self.get_sentence())
        all_sentence.insert(all_sentence.shape[1], 'user', 'unkonwn')
        all_sentence.insert(all_sentence.shape[1], 'label', 0)
        all_sentence.columns = ['sent','user','label']
        label = self.get_label()
        
        for i in range(84000):
            all_sentence.iloc[i,2] = label.iloc[label[label.name == xml_name[i//200][58:-4]].index.tolist()[0],1]
            all_sentence.iloc[i,1] = xml_name[i//200][58:-4]
            
        return all_sentence
    
    def ex_feature(self):
        all_s = self.merge_sent_label()
        embedding = []
        for i in range(84000):
            embedding.append(get_Feature(all_s.iloc[i,0]))
        embedding = pd.DataFrame(embedding)
            
        return embedding
            
    def merge_data_feature(self):
        embed = self.ex_feature()
        all_s = self.merge_sent_label()
        
        columns = all_s.columns.tolist()
        columns.insert(0,'number')
        all_s = all_s.reindex(columns = columns)
        all_s['number'] = np.arange(84000)
        
        columns = embed.columns.tolist()
        columns.insert(0,'number')
        embed = embed.reindex(columns = columns)
        embed['number'] = np.arange(84000)
        
        all_sentence_feature = pd.merge(all_s,embed,how="left")
        
        return all_sentence_feature
    
    
    def sort_data(self):
        data = self.merge_data_feature()
        a = np.array(data).reshape((-1,200,46))
        b = np.random.permutation(a)
        data = pd.DataFrame(b)
        
        train = data.iloc[:72000,:]
        test = data.iloc[72000:,:]
        
        xtrain,xval,ytrain,yval = train_test_split(train,train,test_size=1/6,random_state=1)
        xtest = np.array(test.iloc[:,3:])
        ytest = np.array(test.iloc[:,2])
        
        Xtest = np.zeros((60,200,43))
        Ytest = np.zeros((60))

        for i in range(60):
            for j in range(200):
                Xtest[i,j,:] = xtest[20*i+j,:]
            Ytest[i] = ytest[200*i]
            
        return  xtrain,xval,Xtest,ytrain,yval,Ytest


# In[3]:


class get_Feature():
    def __init__(self,sentence):
        self.sentence = str(sentence)
        self.sent = str(sentence)
        self.deleta_part = ['#USER#','#HASHTAG#','#URL#','\n']
        self.analyzer = SentimentIntensityAnalyzer()
        self.punc = string.punctuation
        self.spec = ['the','that','these','those','this']
        self.pattern = r"(<[A-Za-z0-9]+>)|(</[A-Za-z0-9]+>)|(#[A-Za-z0-9]+#)"
        self.tokenizer = WordPunctTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.tokenizer = BertTokenizerFast.from_pretrained('tokenizer_bert_base')
        self.model = BertForSequenceClassification.from_pretrained('model', num_labels=2,output_hidden_states = False)
        
        #preprocess:usernames, URLs, \n and hashtags removal
        for s in self.deleta_part:
            self.sentence = self.sentence.replace(s,'') 

    
    def compute_upp(self):
        table=str.maketrans('','',self.punc)
        sent = self.sentence.translate(table)
        up_count = np.zeros((1,1))
    
        for i in sent: 
            if i.isupper():
                up_count += 1
        b = len(sent)
        if b == 0:
            a = 0
        else:
            a = up_count/len(sent)
        
        return np.array(a).reshape((1,1))
    
    def compute_number_emoji(self):
        n_emoji = np.array(int(emoji.emoji_count(self.sentence,unique=False))).reshape((1,1))
        
        return np.concatenate((self.compute_upp(),n_emoji),axis = 1)
    
    def compute_spe_punc(self):
        n_exclamation = np.zeros((1,1))
        n_ques = np.zeros((1,1))
        n_c_stop = np.zeros((1,1))
        i_ = str(' ')
        for i in self.sentence:
            if i == str('!'):
                n_exclamation += 1
            elif i == str('?'):
                n_ques += 1
            elif i == str('.') and i_ ==str('.'):
                n_c_stop += 1
            i_ = i

        return np.concatenate((self.compute_number_emoji(),n_exclamation, n_ques, n_c_stop),axis = 1)
    
    def compute_repetitive_punctuations(self):
        s_ = str(' ')
        n_repetitive = np.zeros((1,1))
        for s in str(self.sentence):
            if s in self.punc and s == s_:
                n_repetitive += 1
            s_ = s
            
        return np.concatenate((self.compute_spe_punc(),n_repetitive),axis = 1)
    
    def compute_cohere(self):
        n_spec = np.zeros((1,1))
        sent = self.tokenizer.tokenize(self.sentence)
        b = []
        for word in sent:
            b.append(self.lemmatizer.lemmatize(word))
    
        for s in b:
            if str(s) in self.spec:
                n_spec += 1
        
        return np.concatenate((self.compute_repetitive_punctuations(),n_spec),axis = 1)
        
    
    def compute_sentiment_score(self):
        Sentiment_score = np.array(list(self.analyzer.polarity_scores(self.preprocess()).values())).reshape((1,4))
    
        return np.concatenate((self.compute_cohere(),Sentiment_score),axis = 1)
    
    
    def emoji_to_text(self):
        sent = emoji.demojize(self.sentence, delimiters=("", ""))
        
        return sent
    
    def mean_vec(self):   ##### TODO
        ss = self.emoji_to_text()
        norm_sent = ' '.join(re.sub(self.pattern," ",ss).split())
        norm_sent = [norm_sent]
        me_vec = tokenizer(norm_sent, truncation=True, max_length=32, 
                           padding=True, return_tensors="tf")['input_ids']
        me_vec = me_vec.numpy().reshape((1,-1))
        
        return np.concatenate((self.compute_sentiment_score(),me_vec,np.zeros((1,64-me_vec.shape[1]))),axis = 1)
        
    def vec(self):
        return self.mean_vec()

