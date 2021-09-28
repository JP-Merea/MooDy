from pysentimiento import SentimentAnalyzer, EmotionAnalyzer 
import pandas as pd
import numpy as np 
import math


class analizer:
        def probas_df(df):
                analyzer = SentimentAnalyzer(lang="es")
                txt = df.text.iloc
                probas = [sent.probas for sent in analyzer.predict(txt)]
                probas_neg = [round(output['NEG']*100,2) for output in probas]
                probas_pos = [round(output['POS']*100,2) for output in probas]
                probas_neu = [round(output['NEU']*100,2) for output in probas]
                dict_prob = {'POS': probas_pos, 'NEG': probas_neg, 'NEU': probas_neu}
                df_prob = pd.DataFrame.from_dict(dict_prob)
                return df_prob
        
        def probas_df_emo(df):
                emotion_analyzer = EmotionAnalyzer(lang="es")
                txt = df.text.iloc
                probas = [sent.probas for sent in emotion_analyzer.predict(txt)]
                probas_joy = [round(output['joy']*100,2) for output in probas]
                probas_surprise = [round(output['surprise']*100,2) for output in probas]
                probas_fear = [round(output['fear']*100,2) for output in probas]
                probas_sadness = [round(output['sadness']*100,2) for output in probas]
                probas_anger = [round(output['anger']*100,2) for output in probas]
                probas_disgust = [round(output['disgust']*100,2) for output in probas]
                probas_others = [round(output['others']*100,2) for output in probas]
                dict_prob = {'joy': probas_joy, 'surprise': probas_surprise, 'fear': probas_fear, 'sadness': probas_sadness, 'anger': probas_anger, 'disgust': probas_disgust, 'others': probas_others}
                df_prob = pd.DataFrame.from_dict(dict_prob)
                return df_prob

class model:
        def get_labels(self, threshold, df):
                self.aux = df.groupby('fecha').agg({'fut_bid':'mean', 'bid':'mean', 'text': 'count'}).reset_index()
                self.aux['Label'] = self.aux.fut_bid.map(lambda x: 0 if x >= threshold else 1 if x <= -threshold else 2)
                self.aux['Up'] = self.aux.fut_bid.map(lambda x: 1 if x >= threshold else 0)
                self.aux['Down'] = self.aux.fut_bid.map(lambda x: 1 if x <= -threshold else 0)
                self.aux['Cte'] = self.aux.fut_bid.map(lambda x: 1 if x > -threshold and x < threshold else 0)
                self.aux.rename(columns = {'text':'n_tweets'}, inplace = True)
                return self.aux.sort_values(by="fecha")
        def sigmoid(x):
                s_fun = 1 / (1 + math.exp(-x))
                return s_fun
        
        def tweet_index(self, dfX, a, b, c, d):
                dfX['indice'] = a*dfX['POS'] + b*dfX['NEG'] + c*dfX['retweet_count'] + d
                df2 = dfX.groupby('fecha').agg({'indice':'mean'}).reset_index()
                return df2.sort_values(by="fecha")
        
        def gradient(self, Y, *args):
                df2 = self.tweet_index(*args) 
                df2['prediction'] = df2.indice.map(lambda x: self.sigmoid(x))
                d_a = np.sum(self.tweet_index(args[0],1,0,0,0).indice*(df2.prediction-Y))
                d_b = np.sum(self.tweet_index(args[0],0,1,0,0).indice*(df2.prediction-Y))
                d_c = np.sum(self.tweet_index(args[0],0,0,1,0).indice*(df2.prediction-Y))
                d_d = np.sum(self.tweet_index(args[0],0,0,0,1).indice*(df2.prediction-Y))
                return d_a, d_b, d_c, d_d
        
        def loss(self, Y, *args):
                df2 = self.tweet_index(*args)
                df2['prediction'] = df2.indice.map(lambda x: self.sigmoid(x))
                df2.sort_values(by="fecha")
                return (1/(Y.count()))*np.sum(Y*np.log(df2['prediction']) - (1-Y)*np.log(1-(df2['prediction'])))
        
        def steps(*args, learning_rate = 0.01):
                step_a = args[0]*learning_rate
                step_b = args[1]*learning_rate
                step_c = args[2]*learning_rate
                step_d = args[3]*learning_rate
                return step_a, step_b, step_c, step_d
        
        def update_params(*args):
                a_new = args[0] - args[4]
                b_new = args[1] - args[5]
                c_new = args[2] - args[6]
                d_new = args[3] - args[7]
                return a_new , b_new, c_new, d_new

class train:
        """def __init__(self):
                # Import data only once
                model = model()
                self.sigmoid = model.sigmoid()
                self.tweet_index = model.tweet_index()
                self.gradient = model.gradient()
                self.loss = model.loss()
                self.steps = model.steps()
                self.update_params = model.update_params()"""
                
        def train_model(self,df_blue, label='Up'):
                a = 0
                b = 0
                c = 0
                d = 0
                n_epoch = 200
                loss_history = [self.loss(self.aux[label],df_blue,a,b,c,d)]
                a_history = [a]
                b_history = [b]
                c_history = [c]
                d_history = [d]
                for epoch in range(n_epoch):
                        new_a = self.update_params(a,b,c,d, *self.steps(*self.gradient(self.aux[label],df_blue,a,b,c,d), learning_rate=0.005))[0]
                        new_b = self.update_params(a,b,c,d, *self.steps(*self.gradient(self.aux[label],df_blue,a,b,c,d), learning_rate=0.005))[1]
                        new_c = self.update_params(a,b,c,d, *self.steps(*self.gradient(self.aux[label],df_blue,a,b,c,d), learning_rate=0.005))[2]
                        new_d = self.update_params(a,b,c,d, *self.steps(*self.gradient(self.aux[label],df_blue,a,b,c,d), learning_rate=0.005))[3]
                        a = new_a
                        b = new_b
                        c = new_c
                        d = new_d
                        loss_history.append(self.loss(self.aux[label],df_blue,a,b,c,d))
                        a_history.append(a)
                        b_history.append(b)
                        c_history.append(c)
                        d_history.append(d)
                return a,b,c,d

class predict:
        """def __init__(self):
                # Import data only once
                self.sigmoid = model.sigmoid()
                self.tweet_index = model.tweet_index()
                self.gradient = model.gradient()
                self.loss = model.loss()
                self.steps = model.steps()
                self.update_params = model.update_params()"""
                
        def model_predict(self, df):
                labels = ['Up', 'Down', 'Cte']
                df2 = pd.DataFrame()
                for lab in labels:
                        a, b, c, d = self.train_model(label=lab)
                        best_label = self.tweet_index(df, a, b, c, d)
                        best_label[lab] = best_label.indice.map(lambda x : self.sigmoid(x))
                        df3 = df2.join(best_label[lab], how='outer')
                return df3
