import pandas as pd
import numpy as np
import math

def get_labels(threshold, df):
    aux = df.groupby('fecha').agg({'fut_bid_2':'mean', 'bid':'mean', 'indice': 'mean'}).reset_index()
    aux['Label'] = aux.fut_bid_2.map(lambda x: 0 if x >= threshold else 1 if x <= -threshold else 2)
    return aux

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tweet_index(dfX, a, b, c, d, e, f, g, h):
    dfX['indice'] = (a*dfX['POS'] + b*dfX['NEG'])*(c*dfX['retweet_count'] + d*dfX['user_followers_count'] + e*dfX['favorite_count']
                                                + f*dfX['user_verified'] + g*dfX['user_favourites_count'] + h*dfX['user_friends_count'])
    return dfX

def gradient(Y, *args):
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

def model_predict(df, target):
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