import numpy as np 
import pandas as pd 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Data set
df = pd.read_csv('/content/drive/MyDrive/labeled_tweets.csv')

class days_lag:
    def get_labels2(threshold, df):
        aux = df.groupby('fecha').agg({'fut_bid_2':"mean", 'bid':'mean', 'text': 'count'}).reset_index()
        aux['Label'] = aux.fut_bid_2.map(lambda x: 0 if x >= threshold else 1 if x <= -threshold else 2)
        aux['Up'] = aux.fut_bid_2.map(lambda x: 1 if x >= threshold else 0)
        aux['Down'] = aux.fut_bid_2.map(lambda x: 1 if x <= -threshold else 0)
        aux['Cte'] = aux.fut_bid_2.map(lambda x: 1 if x > -threshold and x < threshold else 0)
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
    def gradient(self, Y, *args):
        df2 = self.tweet_index(*args)
        df2['prediction'] = df2.indice.map(lambda x: self.sigmoid(x))
        d_a = np.sum(self.tweet_index(args[0],1,0,args[3],args[4], args[5],args[6],args[7],args[8]).indice*(df2.prediction-Y))
        d_b = np.sum(self.tweet_index(args[0],0,1,args[3],args[4], args[5],args[6],args[7],args[8]).indice*(df2.prediction-Y))
        d_c = np.sum(self.tweet_index(args[0],args[1],args[2],1,0,0,0,0,0).indice*(df2.prediction-Y))
        d_d = np.sum(self.tweet_index(args[0],args[1],args[2],0,1,0,0,0,0).indice*(df2.prediction-Y))
        d_e = np.sum(self.tweet_index(args[0],args[1],args[2],0,0,1,0,0,0).indice*(df2.prediction-Y))
        d_f = np.sum(self.tweet_index(args[0],args[1],args[2],0,0,0,1,0,0).indice*(df2.prediction-Y))
        d_g = np.sum(self.tweet_index(args[0],args[1],args[2],0,0,0,0,1,0).indice*(df2.prediction-Y))
        d_h = np.sum(self.tweet_index(args[0],args[1],args[2],0,0,0,0,0,1).indice*(df2.prediction-Y))
        return d_a, d_b, d_c, d_d, d_e, d_f, d_g, d_h

    def loss(self, Y, *args):
        df2 = self.tweet_index(*args)
        df2['prediction'] = df2.indice.map(lambda x: self.sigmoid(x))
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
    def train_model(self, target, label='Up'):
        a = 0.1
        b = 0.1
        c = 0.1
        d = 0.1
        e = 0.1
        f = 0.1
        g = 0.1
        h = 0.1
        n_epoch = 200
        loss_history = [self.loss(target[label],df_blue,a,b,c,d,e,f,g,h)]
        a_history = [a]
        b_history = [b]
        c_history = [c]
        d_history = [d]
        e_history = [e]
        f_history = [f]
        g_history = [g]
        h_history = [h]

        for epoch in range(self.n_epoch):
            new_a = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[0]
            new_b = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[1]
            new_c = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[2]
            new_d = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[3]
            new_e = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[4]
            new_f = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[5]
            new_g = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[6]
            new_h = self.update_params(a,b,c,d,e,f,g,h, *self.steps(*self.gradient(target[label],df_blue,a,b,c,d,e,f,g,h), learning_rate=0.003))[7]
            a = new_a
            b = new_b
            c = new_c
            d = new_d
            e = new_e
            f = new_f
            g = new_g
            h = new_h
            loss_history.append(self.loss(target[label],df_blue,a,b,c,d,e,f,g,h))
            a_history.append(a)
            b_history.append(b)
            c_history.append(c)
            d_history.append(d)
            e_history.append(e)
            f_history.append(f)
            g_history.append(g)
            h_history.append(h)
        print(self.loss(target[label],df_blue,a,b,c,d,e,f,g,h))
        print(a,b,c,d,e,f,g,h)
        return a,b,c,d,e,f,g,h
    
    def model_predict(self, df, target):
        labels = ['Up', 'Down', 'Cte']
        df2 = pd.DataFrame()
        for lab in labels:
            a, b, c, d , e, f, g, h= self.train_model(target, label=lab)
            best_label = self.tweet_index(df, a, b, c, d, e, f, g, h)
            best_label[lab] = best_label.indice.map(lambda x : self.sigmoid(x))
            df2 = df2.join(best_label[lab], how='outer')
        df2['predicted_label'] = df2[['Up','Down','Cte']].max(axis=1)
        df2.loc[df2['predicted_label'] == df2['Up'], 'predicted_label'] = 0
        df2.loc[df2['predicted_label'] == df2['Down'], 'predicted_label'] = 1
        df2.loc[df2['predicted_label'] == df2['Cte'], 'predicted_label'] = 2
        df2['predicted_label'] = df2['predicted_label'].astype(int)
        df2['target'] = target['Label']
        df2['succeed'] = df2['target'] == df2['predicted_label']
        return df2

df = pd.read_csv('/content/drive/MyDrive/lstm_input2.csv')
df = df[['indice','bid','Label']]


def subsample_sequence(df, length):
    """
    Given the initial dataframe `df`, return a shorter dataframe sequence of length `length`.
    This shorter sequence should be selected at random.
    """
    
    last_possible = df.shape[0] - length
    
    random_start = np.random.randint(0, last_possible)
    df_sample = df[random_start: random_start+length]
    
    return df_sample

def compute_means(X, df_mean):
    '''utils'''
    # Compute means of X
    means = X.mean()
    
    # Case if ALL values of at least one feature of X are NaN, then reaplace with the whole df_mean
    if means.isna().sum() != 0:
        means.fillna(df_mean, inplace=True)
        
    return means

def split_subsample_sequence(df, length, df_mean=None):
    """Return one single sample (Xi, yi) containing one sequence each of length `length`"""
    features_names = ['indice', 'bid']
    
    # Trick to save time during the recursive calls
    if df_mean is None:
        df_mean = df[features_names].mean()
        
    df_subsample = subsample_sequence(df, length).copy()
    
    # Let's drop any row without a target! We need targets to fit our model
    df_subsample.dropna(how='any', subset=['Label'], inplace=True)
    
    # Create y_sample
    if df_subsample.shape[0] == 0: # Case if there is no targets at all remaining
        return split_subsample_sequence(df, length, df_mean) # Redraw by recursive call until it's not the case anymore
    y_sample = df_subsample[['Label']]#.tail(1).iloc[0]
    
    # Create X_sample
    X_sample = df_subsample[features_names]
    if X_sample.isna().sum().sum() !=0:  # Case X_sample has some NaNs
        X_sample = X_sample.fillna(compute_means(X_sample, df_mean))
        
    return np.array(X_sample), np.array(y_sample)

def get_X_y(df, sequence_lengths):
    '''Return a dataset (X, y)'''
    X, y = [], []

    for length in sequence_lengths:
        xi, yi = split_subsample_sequence(df, length)
        X.append(xi)
        y.append(to_categorical(yi))
        
    return X, y

# Here we define the parameter to generate our train/test sets
train_size = 800
#test_size = round(0.6*train_size)

min_seq_len = 30
max_seq_len = 50

sequence_lengths_train = np.random.randint(low=min_seq_len, high=max_seq_len, size=train_size)
X, y = get_X_y(df, sequence_lengths_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_pad = pad_sequences(X_train, value=-1000., dtype=float, padding='post', maxlen=50)
y_train_pad = pad_sequences(y_train, value=-1000., dtype=float, padding='post', maxlen=50)

X_test_pad = pad_sequences(X_test, value=-1000., dtype=float, padding='post', maxlen=50)
y_test_pad = pad_sequences(y_test, value=-1000., dtype=float, padding='post', maxlen=50)

def init_model():
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=2000, decay_rate=0.5)
    adam = Adam(learning_rate=lr_schedule)
    model = models.Sequential()
    model.add(layers.Masking(mask_value=-1000., input_shape=(50, 2)))
    model.add(layers.GRU(512, return_sequences=True, activation='tanh'))
    model.add(layers.GRU(256, return_sequences=True, activation='tanh'))
    model.add(layers.GRU(128, return_sequences=True, activation='tanh'))
    model.add(layers.GRU(256, return_sequences=True, activation='tanh'))
    model.add(layers.GRU(128, return_sequences=True, activation='tanh',dropout=0.1))
    model.add(layers.GRU(64, return_sequences=True, activation='tanh',dropout=0.2))
    model.add(layers.GRU(32, return_sequences=True, activation='tanh', dropout=0.3 ))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model

from tensorflow.keras.callbacks import EarlyStopping

model = init_model()

es = EarlyStopping(monitor='val_loss', verbose=1, patience=10, restore_best_weights=True)

history = model.fit(X_train_pad, y_train_pad,
            epochs=1000,
            validation_split=0.2, 
            batch_size=32,
            callbacks=[es], 
            verbose=1,
            )

res = model.evaluate(X_test_pad, y_test_pad, verbose=1)
