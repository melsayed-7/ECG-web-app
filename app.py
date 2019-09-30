from flask import Flask, render_template,url_for,request, redirect
from werkzeug import secure_filename
import pandas as pd

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os



import tensorflow as tf
import numpy as np
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import LSTM,Bidirectional, GRU #could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers,regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as K
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam





import pickle as pk
fn = "data_"+"NLRAV"+".pk"
with open(fn, "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)

classes = {0:"Normal",
           1:"Left Bundle Branch block beat",
           2:"Right Bundle Branch block beat",
           3:"Atrial premature beat",
           4:"Premature ventricle contraction"}
#path = Path(__file__).parent
#model_file_url = 'YOUR MODEL.h5 DIRECT / RAW DOWNLOAD URL HERE!'
#model_file_name = 'model'

seqlength = 1000
features = 1
dimout = 1

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

f_size = 300
class_num = 5
lr = 0.005
"""
def getmodel_one():
    model = Sequential()
    
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1), merge_mode='sum', input_shape=(f_size, 1)))
    model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.1), merge_mode='sum'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    #model.compile(loss=f1_loss, #loss=weighted_categorical_crossentropy(weights), 
     #             optimizer=Adam, 
      
#            metrics=['categorical_accuracy', f1_m,precision_m, recall_m]) #(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
    
    
    
    model.add(Dense(32,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid'))
    adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    weights = np.array([64.0, 0.1,])
    model.compile(loss=f1_loss, #loss=weighted_categorical_crossentropy(weights), 
                  optimizer=adam, 
                  metrics=['categorical_accuracy', f1_m,precision_m, recall_m]) #(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
   
    model.add(Dense(class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    print(model.summary())
    return model
    
model = getmodel_one()

model.load_weights("weights_bilstm.h5")

model.predict(X_valid.resize(15709,47,1))
dictionary = {"0":[x_v[0]],"1":[x_v[1]]}

data_frame = pd.DataFrame(dictionary) 
  
# saving the dataframe 
data_frame.to_csv("test_file.csv") 

k = pd.read_csv("test_file.csv")

print(y_v[0])
"""
#############################################


import numpy as np
import pickle as pk
import os, sys
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score 
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam

def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  
K.set_session(get_session())



mode = "NLRAV" # NLRAV / NSVFQ
model_type = "BiLSTM"
save = "result_NLRAV_0_1" # result_NLRAV_0

fn = "data_"+mode+".pk"
with open(fn, "rb") as fp:
    X_train = pk.load(fp)
    Y_train = pk.load(fp)
    X_valid = pk.load(fp)
    Y_valid = pk.load(fp)
    

if model_type != 'Dense':
    X_train = np.expand_dims(X_train, axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)

print(X_train.shape)
print(X_valid.shape)
f_size = X_train.shape[1]
class_num = 5

#============================================#

lr = 0.005
batch_size=128

Y_train = keras.utils.to_categorical(Y_train, num_classes=class_num)

def make_model():
    model = Sequential()
    if model_type == '1D':
        model.add(Conv1D(18, 7, activation='relu', input_shape=(f_size,1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(18, 7, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
    elif model_type == '1D-small':
        model.add(Conv1D(10, 3, activation='relu', input_shape=(f_size,1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(10, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
    elif model_type == '1D-large':
        model.add(Conv1D(50, 13, activation='relu', input_shape=(f_size,1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(50, 13, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
    elif model_type == 'LSTM':
        model.add(LSTM(64, return_sequences=True, dropout=0.1, input_shape=(f_size, 1)))
        model.add(LSTM(32, return_sequences=True, dropout=0.1))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
    elif model_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1), merge_mode='sum', input_shape=(f_size, 1)))
        model.add(Bidirectional(LSTM(32, return_sequences=True, dropout=0.1), merge_mode='sum'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
    elif model_type == 'Dense':
        model.add(Dense(256, input_dim=f_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, input_dim=f_size, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    
    graph = tf.get_default_graph()
    
    return model, graph

model, graph = make_model()




best_SE = 0
best_ACC = 0
best_model = make_model()
patience = 30
pcnt = 0

bin_label = lambda x: min(1,x)

model.load_weights("weights_bilstm.h5")
global model, graph

from keras.backend import clear_session
clear_session()


##############################################

df = pd.DataFrame(X_valid[0,:,0]) 

df_2 = pd.DataFrame(X_valid[6,:,0])
df_3 = pd.DataFrame(X_valid[8,:,0])

df.to_csv("test_file_norm.csv")
df_2.to_csv("test_file_Right Bundle.csv")
df_3.to_csv("Atrial premature beat.csv")


data_test = pd.read_csv("test_file_normal.csv",index_col=0)

data_test = data_test.as_matrix()
data_test = data_test.reshape(1,47,1)
model.predict(data_test)




import io


def pred(data):
    results = model.predict(data)
    print(results)
    return 1


app = Flask(__name__)

model, graph = make_model()
model.load_weights("weights_bilstm.h5")
model._make_predict_function()

@app.route('/')
def home():
	return render_template('home.html',variable = '')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    #file = request.files['ecg_data']
    #data = pd.read_csv(file)
    #print(data)
    f = request.files["ecg_data"]
    #f.save(secure_filename(f.filename))
    #fstring = f.read()
    

    data_test = pd.read_csv(f.filename,index_col=0)
    data_test = data_test.as_matrix()
    data_test = data_test.reshape(1,47,1)
    
    #print(data_test)
    #resultss = model.predict(data_test)
    
    filename = secure_filename(f.filename)
    file_path = os.path.join("data", filename)
    f.save(file_path)
    #print(filename)
    
    #res = pred(data_test)
    #print(res)
    
    res = []
    
    graph = tf.get_default_graph()
    with graph.as_default():
        # perform the prediction
        out = model.predict(data_test)
        for i in range(out.shape[0]):
         out[i] = np.argmax(out[i,:])
         res.append(classes[out[i,0]])
        print("done")
   
    print(res)
   
    return render_template('home.html', variable=res[0])
    
    
    #return redirect(url_for('ress'))
    #return "Hello"



"""
	#Save Model
	joblib.dump(clf, 'model.pkl')
	print("Model dumped!")

	#ytb_model = open('spam_model.pkl', 'rb')
	clf = joblib.load('model.pkl')

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)

	return render_template('result.html', prediction = my_prediction)
"""

if __name__ == '__main__':
	app.run()