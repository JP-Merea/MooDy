# -*- coding: utf-8 -*-
"""Moody Notebook

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14b8GKdD6KKvasJ0Jx9QwB23_nIyvMEw6
"""

pwd

from google.colab import drive
drive.mount('/content/drive/')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/labeled_tweets.csv')

blue = pd.read_csv('/content/drive/MyDrive/dolar_blue_fut.csv')

features = pd.read_csv('/content/drive/MyDrive/features.csv')

import seaborn as sns
sns.heatmap(features.corr())

#Convertir fechas desde object a datetimes
blue['fecha'] = pd.to_datetime(blue['fecha'])
df['created_at'] = pd.to_datetime(df['created_at'])

df = df[df.created_at != 'created_at']

#Cruzar tweeters con información del dolar
df_blue = df.merge(blue, left_on='created_at', right_on='fecha', how='outer')

#Eliminar las fechas sin tweeters
df_blue.dropna(subset = ['created_at'], inplace=True)
print(df_blue.shape)

#Eliminar filas duplicadas
df_blue = df_blue.drop_duplicates()
print(df_blue.shape)

def merge_fechas(df1, df2, criteria = 'forward'):
  df3 = df1.merge(df2, left_on='created_at', right_on='fecha', how='outer')
  df3.dropna(subset = ['created_at'], inplace=True)
  df3 = df3.sort_values(by="created_at")
  if criteria == 'forward':
    df3[['bid','ask']] = df3[['bid','ask']].fillna(method="ffill")
    df3[['bid','ask']] = df3[['bid','ask']].fillna(method="bfill")
  if criteria == 'drop':
    df3.dropna(subset = ['fut_bid'], inplace=True)

df_blue = df_blue.sort_values(by="created_at")
df_blue.head(3)

df_blue[['fecha','fut_bid', 'bid']] = df_blue[['fecha','fut_bid', 'bid']].fillna(method="ffill")
df_blue.dropna(subset = ['fecha'], inplace=True)
df_blue.head(3)

#Desagregar la fecha en DOW, semana, mes y año
df_blue['DOW'] = df_blue['fecha'].dt.weekday+1
df_blue['Week'] = df_blue['fecha'].dt.weekofyear
df_blue['Month'] = df_blue['fecha'].dt.month
df_blue['Year'] = df_blue['fecha'].dt.year
df_blue.head(1)

df_blue['Output_label'] = df_blue.Most_Prob.map({'POS':1, 'NEU':0, 'NEG':0})
df_blue['Output_neg'] = df_blue.Most_Prob.map({'POS':0, 'NEU':0, 'NEG':1})

import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

df_blue['indice'] = df_blue['POS'] + df_blue['NEG']
df2 = df_blue.groupby('fecha').agg({'indice':'mean'}).reset_index()
df2['prediction'] = df2.indice.map(lambda x: sigmoid(x))
(1/(aux.Up.count()))*np.sum(aux['Up']*np.log(df2['prediction']) - (1-aux['Up'])*np.log(1-(df2['prediction'])))
#(1-aux['Up'].to_numpy())*np.log(1-(df2['prediction']))
#np.log(1-(df2['prediction']))
aux['Up']*np.log(df2['prediction']) - (1-aux['Up'])*np.log(1-(df2['prediction']))

def tweet_index(df, a, b, c, d):
  df['indice'] = a*df['POS'] + b*df['NEG'] + c
  df2 = df.groupby('fecha').agg({'indice':'mean'}).reset_index()
  return df2.sort_values(by="fecha")

def gradient(dfX,Y ,a, b, c, d):
    pass  # YOUR CODE HERE
    df2 = tweet_index(dfX, a, b, c, d)
    df2['prediction'] = df2.indice.map(lambda x: sigmoid(x))
    d_a = np.sum(tweet_index(dfX,1,0,0,d).indice*(df2.prediction-Y))
    d_b = np.sum(tweet_index(dfX,0,1,0,d).indice*(df2.prediction-Y))
    d_c = np.sum(tweet_index(dfX,0,0,1,d).indice*(df2.prediction-Y))
    return d_a, d_b, d_c

def loss(dfX, Y, a, b, c, d):
  df2 = tweet_index(dfX, a, b, c, d)
  df2['prediction'] = df2.indice.map(lambda x: sigmoid(x))
  df2.sort_values(by="fecha")
  return (1/(Y.count()))*np.sum(Y*np.log(df2['prediction']) - (1-Y)*np.log(1-(df2['prediction'])))

def steps(d_a,d_b,d_c, learning_rate = 0.01):
    pass  # YOUR CODE HERE
    step_a = d_a*learning_rate
    step_b = d_b*learning_rate
    step_c = d_c*learning_rate
    return step_a, step_b, step_c

def update_params(a, b, c, step_a, step_b, step_c):
    pass  # YOUR CODE HERE
    a_new = a - step_a
    b_new = b - step_b
    c_new = c - step_c
    return a_new , b_new, c_new

print(loss(df_blue, aux.Up, 2, 2, 1, 1.5))
update_params(1,1,1, *steps(*gradient(df_blue, aux['Up'], 1, 1, 1, 1.5), learning_rate=0.005))

np.sum(tweet_index(df_blue,0,1,-1,1.5).indice.to_numpy())

from tqdm.notebook import tqdm

a = 0
b = 0
c = 0
d = 0
n_epoch = 200
loss_history = [loss(df_blue, aux.Down,a,b,c,d)]
a_history = [a]
b_history = [b]
c_history = [c]

for epoch in tqdm(range(n_epoch)):
    new_a = update_params(a,b,c, *steps(*gradient(df_blue, aux.Down, a, b, c, d), learning_rate=0.005))[0]
    new_b = update_params(a,b,c, *steps(*gradient(df_blue, aux.Down, a, b, c, d), learning_rate=0.005))[1]
    new_c = update_params(a,b,c, *steps(*gradient(df_blue, aux.Down, a, b, c, d), learning_rate=0.005))[2]
    a = new_a
    b = new_b
    c = new_c
    loss_history.append(loss(df_blue, aux.Down,a,b,c,d))
    a_history.append(a)
    b_history.append(b)
    c_history.append(c)

a_history[-1],b_history[-1],c_history[-1]

a,b,c = a_history[-1],b_history[-1],c_history[-1]
best_label = tweet_index(df_blue, a, b, c, 1.5)
best_label['predictor'] = best_label.indice.map(lambda x : sigmoid(x))
best_label['prediction'] = best_label.predictor.map(lambda x: 1 if x > 0.366 else 0)
best_label
#def threshold_opt(df):
#  df['prediction'] = df.predictor.map(lambda x: 1 if x > threshold else 0)
df_validación = aux.merge(best_label, on='fecha', how='inner')
df_validación['correct'] = df_validación['Down'] ==  df_validación['prediction']
df_validación.correct.value_counts()

aux = df_blue.groupby('fecha').agg({'Output_label':'sum','Output_neg':'sum', 'fut_bid':'mean', 'bid':'mean', 'text': 'count'}).reset_index()
aux['Up'] = aux.fut_bid.map(lambda x: 1 if x >= 0.8 else 0)
aux['Down'] = aux.fut_bid.map(lambda x: 1 if x <= -0.8 else 0)
aux['Cte'] = aux.fut_bid.map(lambda x: 1 if x > -0.8 and x < 0.8 else 0)
#df_dia = aux.merge(blue[['fecha','bid','fut_bid']], left_on='fecha', right_on='fecha', how='outer')
#df_dia.dropna(subset = ['Output_label'], inplace=True)
aux.rename(columns = {'text':'n_tweets'}, inplace = True)
aux.sort_values(by="fecha")
aux.head(15)
#aux[aux['Output_label']<-0.3]

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(aux.corr()), aux.corr()

fig, axs = plt.subplots(4, 1, figsize=(18,10))
axs[0].plot(aux.Output_label, c='red')
axs[1].plot(-aux.Output_neg, c='purple')
axs[2].plot(aux.bid)
axs[3].plot(aux.fut_bid, c='green')

aux.to_csv('output_df.csv')

!cp output_df.csv "drive/My Drive/"

AVAILABLE_LANGS = ['es', 'en']
import emoji

def check_valid_lang(lang):
    if lang not in AVAILABLE_LANGS:
        raise Exception(f'There is no stopwords file for {lang}.') 

def read_stopwords(lang):
    '''Reads stopword file, returns a list of stopwords.'''
    check_valid_lang(lang)
    with open(f'../stopwords/stopwords_{lang}.txt', 'r') as fi:
        stop_words = fi.read()
    return stop_words.split()

def remove_emojis(data):
    #return re.sub(r':[^: ]+?:', '', tweet)
    #return emoji.get_emoji_regexp().sub(r'', tweet.decode('utf8'))
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

import string
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def prep(text):
    text = remove_emojis(text).strip()
    text = re.sub(r'http\S+', '', text).strip()
    text = re.sub(r'#', '', text).strip()
    text = re.sub(r'@\S+', '', text).strip()
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    text = text.lower()
    text = ''.join(word for word in text if not word.isdigit())
    stop_words = set(stopwords.words('spanish'))
    word_tokens = word_tokenize(text)
    text = ' '.join(w for w in word_tokens if not w in stop_words)
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens]
    text = ' '.join(w for w in lemmatized)
    return text.strip()

df_blue['text_prep'] = df_blue.text.map(lambda x : prep(x))

texto = '#lado b covid #pasando @patologías ✍️ gustavo sar httpshola que suba el dolar'
prep(texto)
