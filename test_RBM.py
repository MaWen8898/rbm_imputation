import os
from rbm import RBM
from au import AutoEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep
import numpy as np

def min_max_scale(X_train, X_test):
    preprocessor = prep.MinMaxScaler().fit(np.concatenate((X_train, X_test), axis=0))
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test 

#get next_batch 
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#flags indicate some parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_integer('epochs', 100, 'The number of training epochs')
flags.DEFINE_integer('batchsize', 30, 'The batch size')
flags.DEFINE_boolean('restore_rbm', False, 'Whether to restore the RBM weights or not.')

# Load Data
import pandas as pd
import xlrd
train = 'smart.xlsx'
train_xl = pd.ExcelFile(train)
train_data= train_xl.parse('TestSheet')
Y=train_data.iloc[:,-1]
X=train_data.iloc[:,:-1]
trX, teX, trY,teY = train_test_split(X, Y,test_size=0.2,random_state=0)
trX, teX = min_max_scale(trX, teX)

import random
import numpy as np

iterations = int(len(trX) / FLAGS.batchsize)
print('training RBM')
# RBMs(smart has 27 variables and 3874 samples)
rbm = RBM(27, 27, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.05)    
                   
import time
start_time = time.time()
    # Train First RBM
from sklearn.utils import shuffle

for i in range(FLAGS.epochs):
    for j in range(iterations):
        batch_xs, batch_ys = next_batch(FLAGS.batchsize,trX,trY)
        """
        sample = np.random.randint(len(trX), size=FLAGS.batchsize)
        batch_xs = trX[sample][:]
        batch_ys = trY[sample][:]
        """
        
        rbm.partial_fit(shuffle(batch_xs))
print("--- %s seconds ---" % (time.time() - start_time))
h_prob=rbm.transform(trX)

from sklearn import linear_model, metrics

logistic = linear_model.LogisticRegression(C=100.0)
logistic.fit(h_prob,trY)
print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
    teY,
    logistic.predict(teX))))


"""
writer = pd.ExcelWriter('recon.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
pd.DataFrame (rbmobject1.reconstruct(trX)).to_excel(writer, sheet_name='TestSheet')
writer.save()

writer = pd.ExcelWriter('recon2.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
pd.DataFrame (rbmobject2.reconstruct(trX)).to_excel(writer, sheet_name='TestSheet')
writer.save()

writer = pd.ExcelWriter('recon3.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
pd.DataFrame (rbmobject3.reconstruct(trX)).to_excel(writer, sheet_name='TestSheet')
writer.save()"""
