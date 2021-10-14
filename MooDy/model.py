import numpy as np
import pandas as pd 
from index import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from datos import get_data, get_clean_data


#df = pd.read_csv('/content/drive/MyDrive/lstm_input2.csv')
#df = df[['indice','bid','Label']]

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

def init_model():
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=2000, decay_rate=0.5)
    adam = Adam(learning_rate=lr_schedule)
    model = models.Sequential()
    model.add(layers.Masking(mask_value=-1000., input_shape=(30, 1)))
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

def train_split(df): 
    # Here we define the parameter to generate our train/test sets
    train_size = 600
    #test_size = round(0.6*train_size)
    min_seq_len = 20
    max_seq_len = 30
    sequence_lengths_train = np.random.randint(low=min_seq_len, high=max_seq_len, size=train_size)
    X, y = get_X_y(df, sequence_lengths_train)
    X_train, y_train = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, y_train

def train_model(X_train, y_train):
    X_train_pad = pad_sequences(X_train, value=-1000., dtype=float, padding='post', maxlen=30)
    y_train_pad = pad_sequences(y_train, value=-1000., dtype=float, padding='post', maxlen=30)
    model = init_model()
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=10, restore_best_weights=True)
    history = model.fit(X_train_pad, y_train_pad,
            epochs=1000,
            validation_split=0.2, 
            batch_size=32,
            callbacks=[es], 
            verbose=1,
            )
    return history

if __name__ == '__main__':
    df, blue = get_data()
    df = get_clean_data(df, blue)
    target= get_labels(0, df)
    a, b, c, d, e, f, g, h = train_model_index(df, target, label='Up')
    dfX = tweet_index(df, a, b, c, d, e, f, g, h)
    df_deep= get_labels(0, dfX)