"""train - prediction of our model-"""""

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