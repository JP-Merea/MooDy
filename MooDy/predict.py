"""train - prediction of our model-"""""
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.layers import Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import get_X_y
from sklearn.model_selection import train_test_split
from index import tweet_index
from datos import get_data, get_clean_data


df1, df2 = get_data()   
df_blue = get_clean_data(df1,df2)
#-----
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
    #model.add(Attention(32))
    #model.add(Flatten())  
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model

#-------
def train_split(): 
    # Here we define the parameter to generate our train/test sets
    train_size = 600
    #test_size = round(0.6*train_size)

    min_seq_len = 20
    max_seq_len = 30

    sequence_lengths_train = np.random.randint(low=min_seq_len, high=max_seq_len, size=train_size)
    X, y = get_X_y(df, sequence_lengths_train)

    X_train, y_train = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, y_train

#-------

def train_model():
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