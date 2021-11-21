from MooDy.index import tweet_index
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from MooDy import index

def model_load():
    filepath = 'gru_model.h5'
    model = load_model(filepath, custom_objects=None, compile=True, options=None)
    return model

def predict_graphic(df):
    model = model_load()
    df = df[['indice','bid','Label']]
    pred_down = []
    for i in range(1,20):
        df_sample = df[-50+i:-20+i]['indice']
        X_predict = np.array(df_sample).reshape((1,30))
        pred_down.append(model.predict(X_predict)[0][0])
    df_sample = df[-50+20:]['indice']
    X_predict = np.array(df_sample).reshape((1,30))
    pred_down.append(model.predict(X_predict)[0][0])
    pred_up = [1-p for p in pred_down]
    arr = np.array([pred_up,pred_down]).T
    df_plot = pd.DataFrame(arr, columns=['a','b'])
    return df_plot

def predict_value(df, predict):
    a, b, c, d, e, f, g, h = -3.9468623263798803, 1.1376029320884422, 0.0983053572559721, 0.06670780452645171, 0.10515254158894936, 2.639352389385343, 0.23819587655091048, 0.05692568577444351
    model = model_load()
    df_sample = df[-29:]['indice']
    include = index.tweet_index(predict, a, b, c, d, e, f, g, h)
    df_sample = df_sample.append(include['indice'])
    X_predict = np.array(df_sample).reshape((1,30))
    return 1-model.predict(X_predict)[0][0], model.predict(X_predict)[0][0]

if __name__ == '__main__':
    df = pd.read_csv('raw_data/lstm_input4.csv')
    df = df[['indice','bid','Label']]
    df[['0','1']] = to_categorical(df['Label'])
