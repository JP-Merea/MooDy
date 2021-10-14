"""In this file we are going to make an index of a logit function. The objective was to make an index that can weigh our parameters and be more representive of 
our dataset"""

import pandas as pd
import numpy as np
from datos import get_data, get_clean_data


# df1, df2 = get_data()   
# df_blue = get_clean_data(df1,df2)

def get_labels(threshold, df):
    aux = df.groupby("fecha").agg({'fut_bid_2':'mean', 'bid':'mean', 'indice': 'mean'}).reset_index()
    aux['Label'] = aux.fut_bid_2.map(lambda x: 0 if x >= threshold else 1 if x <= -threshold else 2)
    return aux

def sigmoid(x):
    """we are going to use to change a linear model to a logit one"""
    return 1 / (1 + np.exp(-x))

def tweet_index(dfX, a, b, c, d, e, f, g, h):
    
    dfX['indice'] = (a*dfX['POS'] + b*dfX['NEG'])*(c*dfX['retweet_count'] + d*dfX['user_followers_count'] + e*dfX['favorite_count']
                                                + f*dfX['user_verified'] + g*dfX['user_favourites_count'] + h*dfX['user_friends_count'])
    return dfX

def gradient(Y, *args):
    """We are going to make a function that calculate the gradient, that we are going to use"""
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
    """We calculate the loss making our own function"""
    df2 = tweet_index(*args)
    df2['prediction'] = df2.indice.map(lambda x: sigmoid(x))
    df2.sort_values(by="fecha")
    return -((1/(Y.count()))*np.sum(Y*np.log(df2['prediction']) + (1-Y)*np.log(1-(df2['prediction']))))

def steps(*args, learning_rate = 0.01):
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
    a_new = args[0] - args[8]
    b_new = args[1] - args[9]
    c_new = args[2] - args[10]
    d_new = args[3] - args[11]
    e_new = args[4] - args[12]
    f_new = args[5] - args[13]
    g_new = args[6] - args[14]
    h_new = args[7] - args[15]
    return a_new , b_new, c_new, d_new, e_new, f_new, g_new, h_new

def train_model(df, target, label='Up'):
    """We train our model using the functions we have create before"""
    a = 0.1
    b = 0.1
    c = 0.1
    d = 0.1
    e = 0.1
    f = 0.1
    g = 0.1
    h = 0.1
    n_epoch = 200
    loss_history = [loss(target[label],df,a,b,c,d,e,f,g,h)]
    a_history = [a]
    b_history = [b]
    c_history = [c]
    d_history = [d]
    e_history = [e]
    f_history = [f]
    g_history = [g]
    h_history = [h]

    for epoch in range(n_epoch):
        new_a = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[0]
        new_b = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[1]
        new_c = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[2]
        new_d = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[3]
        new_e = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[4]
        new_f = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[5]
        new_g = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[6]
        new_h = update_params(a,b,c,d,e,f,g,h, *steps(*gradient(target[label],df,a,b,c,d,e,f,g,h), learning_rate=0.003))[7]
        a = new_a
        b = new_b
        c = new_c
        d = new_d
        e = new_e
        f = new_f
        g = new_g
        h = new_h
        loss_history.append(loss(target[label],df,a,b,c,d,e,f,g,h))
        a_history.append(a)
        b_history.append(b)
        c_history.append(c)
        d_history.append(d)
        e_history.append(e)
        f_history.append(f)
        g_history.append(g)
        h_history.append(h)
    print(loss(target[label],df,a,b,c,d,e,f,g,h))
    print(a,b,c,d,e,f,g,h)
    return a,b,c,d,e,f,g,h

def model_predict(df, target):
    """We make our first prediction"""
    labels = ['Up', 'Down', 'Cte']
    df2 = pd.DataFrame()
    for lab in labels:
        a, b, c, d , e, f, g, h= train_model(target, label=lab)
        best_label = tweet_index(df, a, b, c, d, e, f, g, h)
        best_label[lab] = best_label.indice.map(lambda x : sigmoid(x))
        df2 = df2.join(best_label[lab], how='outer')
    df2['predicted_label'] = df2[['Up',"Down",'Cte']].max(axis=1)
    df2.loc[df2['predicted_label'] == df2['Up'], 'predicted_label'] = 0
    df2.loc[df2['predicted_label'] == df2['Down'], 'predicted_label'] = 1
    df2.loc[df2['predicted_label'] == df2['Cte'], 'predicted_label'] = 2
    df2['predicted_label'] = df2['predicted_label'].astype(int)
    df2['target'] = target['Label']
    df2['succeed'] = df2['target'] == df2['predicted_label']
    return df2
