import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import os.path




filepath = 'streamlit/saved_model.pb'

if __name__ == '__main__':
    print( os.path.isfile(filepath)) 
    model = tf.keras.models.load_model(filepath, custom_objects=None, compile=True, options=None)
    print(model)
    X = np.array([[0.164680]])
    print(X)
    X_test_pad = pad_sequences(X, value=-1000., dtype=float, padding='post', maxlen=30)
    print( model.predict(X))
