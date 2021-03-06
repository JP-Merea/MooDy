#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pandas as pd
import numpy as np

# Data set
df = pd.read_csv('/content/drive/MyDrive/labeled_tweets.csv')

class days_lag:
    def get_labels2(threshold, df):
        aux = df.groupby('fecha').agg({'fut_bid_2':'mean', 'bid':'mean', 'text': 'count'}).reset_index()
        aux['Label'] = aux.fut_bid_2.map(lambda x: 0 if x >= threshold else 1 if x <= -threshold else 2)
        aux['Up'] = aux.fut_bid_2.map(lambda x: 1 if x >= threshold else 0)
        aux['Down'] = aux.fut_bid_2.map(lambda x: 1 if x <= -threshold else 0)
        aux['Cte'] = aux.fut_bid_2.map(lambda x: 1 if x > -threshold and x < threshold else 0)
        #df_dia = aux.merge(blue[['fecha','bid','fut_bid']], left_on='fecha', right_on='fecha', how='outer')
        #df_dia.dropna(subset = ['Output_label'], inplace=True)
        aux.rename(columns = {'text':'n_tweets'}, inplace = True)
        return aux.sort_values(by="fecha")

# Logit (index)
class logit_1:
    
    def __init__(self):
        pass
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tweet_index(dfX, a, b, c, d, e, f, g, h):
        dfX['indice'] = (a*dfX['POS'] + b*dfX['NEG'])*(c*dfX['retweet_count'] + d*dfX['user_followers_count'] + e*dfX['favorite_count']
                                                        + f*dfX['user_verified'] + g*dfX['user_favourites_count'] + h*dfX['user_friends_count'])
        df2 = dfX.groupby('fecha').agg({'indice':'mean'}).reset_index()
        return df2.sort_values(by="fecha")

    #def gradient(Y, dfX, a, b, c, d):
    def gradient(Y, *args):
        pass  # YOUR CODE HERE
        df2 = tweet_index(*args)
        df2['prediction'] = df2.indice.map(lambda x: sigmoid(x))
        d_a = np.sum(tweet_index(args[0],1,0,args[3],args[4], args[5],args[6],args[7],args[8]).indice*(df2.prediction-Y))
        d_b = np.sum(tweet_index(args[0],0,1,args[3],args[4], args[5],args[6],args[7],args[8]).indice*(df2.prediction-Y))
        d_c = np.sum(tweet_index(args[0],args[1],args[2],1,0,0,0,0,0).indice*(df2.prediction-Y))
        d_d = np.sum(tweet_index(args[0],args[1],args[2],0,1,0,0,0,0).indice*(df2.prediction-Y))
        d_e = np.sum(tweet_index(args[0],args[1],args[2],0,0,1,0,0,0).indice*(df2.prediction-Y))
        d_f = np.sum(tweet_index(args[0],args[1],args[2],0,0,0,1,0,0).indice*(df2.prediction-Y))
        d_g = np.sum(tweet_index(args[0],args[1],args[2],0,0,0,0,1,0).indice*(df2.prediction-Y))
        d_h = np.sum(tweet_index(args[0],args[1],args[2],0,0,0,0,0,1).indice*(df2.prediction-Y))
        return d_a, d_b, d_c, d_d, d_e, d_f, d_g, d_h

    def loss(Y, *args):
        df2 = tweet_index(*args)
        df2['prediction'] = df2.indice.map(lambda x: sigmoid(x))
        df2.sort_values(by="fecha")
        return -((1/(Y.count()))*np.sum(Y*np.log(df2['prediction']) + (1-Y)*np.log(1-(df2['prediction']))))

    def steps(*args, learning_rate = 0.01):
        pass  # YOUR CODE HERE
        step_a = args[0]*learning_rate
        step_b = args[1]*learning_rate
        step_c = args[2]*learning_rate
        step_d = args[3]*learning_rate
        step_e = args[4]*learning_rate
        step_f = args[5]*learning_rate
        step_g = args[6]*learning_rate
        step_h = args[7]*learning_rate
        return step_a, step_b, step_c, step_d, step_e, step_f, step_g, step_h

    def update_params(*args):
        pass  # YOUR CODE HERE
        a_new = args[0] - args[8]
        b_new = args[1] - args[9]
        c_new = args[2] - args[10]
        d_new = args[3] - args[11]
        e_new = args[4] - args[12]
        f_new = args[5] - args[13]
        g_new = args[6] - args[14]
        h_new = args[7] - args[15]
        return a_new , b_new, c_new, d_new, e_new, f_new, g_new, h_new

# Model
class model:
    def train_model(target, label='Up'):
        a = 0.1
        b = 0.1
        c = 0.1
        d = 0.1
        e = 0.1
        f = 0.1
        g = 0.1
        h = 0.1
        n_epoch = 200
        loss_history = [loss(target[label],df_blue,a,b,c,d,e,f,g,h)]
        a_history = [a]
        b_history = [b]
        c_history = [c]
        d_history = [d]
        e_history = [e]
        f_history = [f]
        g_history = [g]
        h_history = [h]

        for epoch in range(n_epoch):
            new_a = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[0]
            new_b = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[1]
            new_c = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[2]
            new_d = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[3]
            new_e = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[4]
            new_f = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[5]
            new_g = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[6]
            new_h = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[7]
            a = new_a
            b = new_b
            c = new_c
            d = new_d
            e = new_e
            f = new_f
            g = new_g
            h = new_h
            loss_history.append(loss(target[label],df_blue,a,b,c,d,e,f,g,h))
            a_history.append(a)
            b_history.append(b)
            c_history.append(c)
            d_history.append(d)
            e_history.append(e)
            f_history.append(f)
            g_history.append(g)
            h_history.append(h)
        print(loss(target[label],df_blue,a,b,c,d,e,f,g,h))
        print(a,b,c,d,e,f,g,h)
        return a,b,c,d,e,f,g,h
    
    def model_predict(df, target):
        labels = ['Up', 'Down', 'Cte']
        df2 = pd.DataFrame()
        for lab in tqdm(labels):
            a, b, c, d , e, f, g, h= train_model(target, label=lab)
            best_label = tweet_index(df, a, b, c, d, e, f, g, h)
            best_label[lab] = best_label.indice.map(lambda x : sigmoid(x))
            df2 = df2.join(best_label[lab], how='outer')
        df2['predicted_label'] = df2[['Up','Down','Cte']].max(axis=1)
        df2.loc[df2['predicted_label'] == df2['Up'], 'predicted_label'] = 0
        df2.loc[df2['predicted_label'] == df2['Down'], 'predicted_label'] = 1
        df2.loc[df2['predicted_label'] == df2['Cte'], 'predicted_label'] = 2
        df2['predicted_label'] = df2['predicted_label'].astype(int)
        df2['target'] = target['Label']
        df2['succeed'] = df2['target'] == df2['predicted_label']
        return df2