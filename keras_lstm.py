#!/usr/bin/python
import time
import random
import subprocess
import sys
from IPython.core.debugger import Tracer
import os

home_dir = '/home/c2tao/'
scp_file = os.path.join(home_dir, 'globalPhone_feat/Czech/train.39.cmvn.scp')
print scp_file
shuf_scp_file = os.path.join(home_dir, 'globalPhone_feat/Czech/train.39.cmvn.shuf.scp')
answer_file = 'answer.label'
shuf_answer_file = 'answer.shuf.label'
shuf_command = 'paste -d \'@\' {} {} | shuf | awk -v FS="@" \'{{ print $1 > \"{}\" ; print $2 > \"{}\" }}\''.format(scp_file,answer_file,shuf_scp_file,shuf_answer_file)
wrd_file = 'label_set'
batch_size = 256
feature_dim = 39
voc_dim =  43 #The length of label_set

def file_len(filename):
    with open(scp_file,'r') as f:
        count = 0;
        for l in f.readlines():
                count += 1
    return count

def max_mfcc_len(scp_file):
    start_time = time.time()
    feat_proc = subprocess.Popen(['feat-to-len scp:{} ark,t:- 2>/dev/null'.format(scp_file)],stdout=subprocess.PIPE,shell=True)
    max_feature_count = -1
    while True:
        line = feat_proc.stdout.readline()
        line = line.rstrip('\n')
        if line == '':
            feat_proc.terminate()
            print "max_len =",max_feature_count
            elapsed_time = time.time() - start_time
            print "cal mfcc len takes {} secs".format( elapsed_time)
            return max_feature_count

        else:
            feature_count = int((line.split())[1])
            if feature_count > max_feature_count:
                max_feature_count = feature_count

def get_feature(raw_content):
    features = raw_content.split('\n')
    features = features[1:-1] #filt out '  [' and the next uttid
    features[-1] = (features[-1])[:-1] #filt out ']'
    for idx, feat in enumerate(features):
        features[idx] = [float(s) for s in feat.split() ]

    return features

def RNN_sequence_data_gen(shuf_scp_file,shuf_answer_file,max_sequence_len): #For RNN, generate batch_size of time sequence data
    feat_proc = subprocess.Popen(['copy-feats scp:{} ark,t:- 2>/dev/null'.format(shuf_scp_file)],stdout=subprocess.PIPE,shell=True)
    ans_file = open(shuf_answer_file,'r')

    start = False
    utt_count = 0

    X_sequence = np.zeros((1,max_sequence_len,feature_dim))
    X = np.zeros((max_sequence_len,feature_dim))

    Y_sequence = np.zeros((1,1))
    Y = np.zeros((1))

    feature_idx = 0
    processed_uttID = ''
    while True:
        if utt_count >= batch_size:
            X_sequence = X_sequence[1:,:,:]
            Y_sequence = Y_sequence[1:]
            yield (X_sequence,utt_count,Y_sequence)
            X_sequence = np.zeros((1,max_sequence_len,feature_dim))
            Y_sequence = np.zeros((1,1))
            utt_count = 0

        line = feat_proc.stdout.readline()
        line = line.rstrip('\n')
        if line == '':
            #end of the ark

            #shuffle scp file
            #close popoen and reopen a new one
            feat_proc.terminate()
            ans_file.close()
            yield (X_sequence,utt_count , Y_sequence )
            break

        if '[' in line :
            assert(start == False)
            start = True
            processed_uttID = (line.split())[0]
            ans_line = ans_file.readline()
            if not ans_line:
                print "error in read ans_file, exit..."
            exit(1)
            ans_line = ans_line.rstrip('\n')
            ans_seg = ans_line.split()
            assert(ans_seg[0] == processed_uttID)

            #Process Y data
            Y = np.array([int(ans_seg[-1])])
            Y = np.expand_dims(Y,axis=0)
            Y_sequence = np.append(Y_sequence,Y,axis=0)

            continue
        
        if start == True and ']' not in line:
            feature = np.array([float(s) for s in line.split()])
            X[feature_idx,:] = feature
            feature_idx += 1
            continue

        if ']' in line:
            #features
            feature = np.array([float(s) for s in (line[:-1]).split()])
            X[feature_idx,:] = feature
            feature_idx += 1
            start = False
            X = np.expand_dims(X,axis=0)
            X_sequence = np.append(X_sequence,X,axis=0)
            X = np.zeros((max_sequence_len,feature_dim))
            feature_idx = 0
            utt_count += 1

################################    Script  #############################################
from keras.layers import containers, Dropout, LSTM
from keras.layers.core import  AutoEncoder, Dense, TimeDistributedDense
from keras.optimizers import RMSprop
from keras import models
from keras.utils import generic_utils
import numpy as np

start_time = time.time()

#get the counts of data
print "get utt_counts"
utt_counts = file_len(scp_file)
print "get max sequence length"
#max_sequence_len = max_mfcc_len(scp_file)
max_sequence_len = 3979 #For Czech corpus

print "Shuf data..."
start_time = time.time()
subprocess.call(shuf_command,shell=True)
print "Elapsed time for shuf = ",time.time() - start_time

#get generator
generator = RNN_sequence_data_gen(shuf_scp_file,shuf_answer_file,max_sequence_len)

#for g in generator:
#    print g[0].shape, g[1], g[2]


print "Building models"
#Models
lstm_model = models.Sequential()

#lstm_model.add(embedding)
lstm_model.add(LSTM(output_dim=128,activation='sigmoid',return_sequences=False,inner_activation='hard_sigmoid',input_shape=(max_sequence_len, feature_dim)))
lstm_model.add(Dense(32, init='uniform'))
lstm_model.add(Dense(1))
progbar = generic_utils.Progbar(utt_counts,width = 60)

# training the autoencoder:
rms_prop = RMSprop(lr=0.00008)
lstm_model.compile(optimizer=rms_prop,loss='mse')

#TODO Write scripts for saving model's weghts

epoch_num = 20
epoch = 1
while epoch <= epoch_num:
    start_time = time.time()
    print ""
    print "Epoch: ",epoch
    counts = 0
    progbar = generic_utils.Progbar(utt_counts,width = 60)
    while True:
        (X,processed_utt,Y) = generator.next()
        counts += processed_utt
         
        #loss = autoencoder_model.train_on_batch(X,X[:,0,:])
        loss = lstm_model.train_on_batch(X,Y)
        progbar.add(processed_utt, values=[("train loss", np.asarray(loss)[0])])
        if(counts >= utt_counts):
            assert(counts == utt_counts)
            break
        
    elapsed_time = time.time() - start_time
    (m,s) = divmod(elapsed_time,60)
    (h,m) = divmod(m,60)
    
    print "Elapsed time for epoch %2d = {} hours {} mins {} secs".format(h,m,s) %(epoch)

    epoch += 1
    #shuffle scp
    subprocess.call(shuf_command,shell=True)
    generator = RNN_sequence_data_gen(shuf_scp_file,shuf_answer_file,max_sequence_len)

elapsed_time = time.time() - start_time
(m,s) = divmod(elapsed_time,60)
(h,m) = divmod(m,60)

print "Elapsed time for whole training process = {} hours {} mins {} secs".format(h,m,s)
