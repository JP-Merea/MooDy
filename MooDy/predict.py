import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 


filepath = '../streamlit/gru_model.h5'

if __name__ == '__main__':
    model = load_model (filepath, custom_objects=None, compile=True, options=None)
    X = np.array([[0.164680]])
    X_test_pad = pad_sequences(X, value=-1000., dtype=float, padding='post', maxlen=30)
    result = model.predict(X_test_pad)
    print(result[0],result[1])
