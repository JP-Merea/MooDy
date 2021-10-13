"""train - prediction of our model-"""""
from tensorflow.keras.callbacks import EarlyStopping
#
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
  labels = ['Label']
  df2 = pd.DataFrame()
  for lab in labels:
    a, b, c, d , e, f, g, h= train_model(target, label=lab)
    best_label = tweet_index(df, a, b, c, d, e, f, g, h)
    best_label[lab] = best_label.indice.map(lambda x : sigmoid(x))
    df2 = df2.join(best_label[lab], how='outer')
  df2['predicted_label'] = df2[['lab']]
  df2['predicted_label'] = df2['predicted_label'].astype(int)
  df2['target'] = target['Label']
  df2['succeed'] = df2['target'] == df2['predicted_label']
  return df2

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

init_model().summary()

#-------


model = init_model()

es = EarlyStopping(monitor='val_loss', verbose=1, patience=10, restore_best_weights=True)

history = model.fit(X_train_pad, y_train_pad,
           epochs=1000,
           validation_split=0.2, 
           batch_size=32,
           callbacks=[es], 
           verbose=1,
           )