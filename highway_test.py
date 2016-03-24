from keras.layers import containers, Dropout, LSTM
import theano
import theano.tensor as T
from keras.optimizers import RMSprop
from keras import models
from keras.utils import generic_utils
import numpy as np
from keras.layers.core import  Activation, AutoEncoder, Dense, TimeDistributedDense, TimeDistributedMerge, Merge, Lambda
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential, Graph
def test_sequential():
    model = Sequential()
    model.add(Dense(10, input_dim=13))
    model.add(Dense(13))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    X_train = np.random.rand(64,13)
    model.fit(X_train, X_train, nb_epoch=100,batch_size=16, verbose=1)

def test_lstm():
    model = Sequential()
    model.add(TimeDistributedDense(10, input_dim=13))
    model.add(LSTM(7, return_sequences=True))
    model.add(TimeDistributedDense(13))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    X_train = np.random.rand(64,10,13)
    model.fit(X_train, X_train, nb_epoch=100,batch_size=16, verbose=1)

def test_custom_lstm():
    from keras.layers.recurrent import Highway_LSTM, Highway_GRU, Highway_SimpleRNN
    from keras.layers.recurrent import Residual_LSTM, Residual_GRU, Residual_SimpleRNN

    rnn = LSTM
    rnn = GRU
    rnn = SimpleRNN

    rnn = Highway_LSTM
    rnn = Highway_GRU
    #rnn = Highway_SimpleRNN

    rnn = Residual_LSTM
    rnn = Residual_GRU
    #rnn = Residual_SimpleRNN

    model = Sequential()
    model.add(TimeDistributedDense(10, input_dim=13))
    model.add(rnn(10, return_sequences=True))
    model.add(TimeDistributedDense(13))
    model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adam')
    X_train = np.random.rand(64,10,13)
    model.fit(X_train, X_train, nb_epoch=100,batch_size=16, verbose=1)

def test_graph():
    graph = Graph()
    graph.add_input(name='input', input_shape=(32,))
    graph.add_node(Dense(16), name='dense1', input='input')
    graph.add_node(Dense(32), name='dense2', input='input')
    graph.add_node(Dense(32), name='dense3', input='dense1')
    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')

    graph.compile(optimizer='rmsprop', loss={'output1':'mse', 'output2':'mse'})

    X_train = np.random.rand(64, 32)
    history = graph.fit({'input':X_train, 'output1':X_train, 'output2':X_train}, nb_epoch=10)

def test_custom_objective():

    graph = Graph()
    graph.add_input(name='input', input_shape=(32,))
    graph.add_node(Dense(16), name='dense1', input='input')
    graph.add_node(Dense(32), name='dense2', input='input')
    graph.add_node(Dense(32), name='dense3', input='dense1')
    graph.add_output(name='output1', input='dense2')
    graph.add_output(name='output2', input='dense3')


    X_train = np.random.rand(64, 32)


    import theano
    import theano.tensor as T

    epsilon = 1.0e-9
    def custom_objective(y_true, y_pred):
        '''Just another crossentropy'''
        y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        cce = T.nnet.categorical_crossentropy(y_pred, y_true)
        return cce

    graph.compile(optimizer='rmsprop', loss={'output1':'mse', 'output2':custom_objective}, loss_weights={'output1':1,'output2':-0.001})
    history = graph.fit({'input':X_train, 'output1':X_train, 'output2':X_train}, nb_epoch=10)
    print graph.inputs['input']
    print graph.nodes['dense1']




def test_graph_merge():
    graph = Graph()
    graph.add_input(name='input', input_shape=[10,39])
    graph.add_node(TimeDistributedDense(10), name='proj', input='input')
    graph.add_node(LSTM(10, return_sequences=True), name='lstm', input='proj')
    graph.add_node(TimeDistributedDense(10), merge_mode = 'sum', name = 'merge', inputs=['proj','lstm'])
    graph.add_node(TimeDistributedDense(39), name='proj2', input='merge')
    graph.add_output(name='output', input='proj2')
    
    X_train = np.random.rand(64,10, 39)
    graph.compile(optimizer='rmsprop', loss={'output':'mse' })
    history = graph.fit({'input':X_train, 'output':X_train}, nb_epoch=10)




def test_lambda_join():
    def custom_fun(X):
        print X.keys()
        return X['proj']+X['lstm']

    graph = Graph()
    graph.add_input(name='input', input_shape=[10, 39])
    graph.add_node(TimeDistributedDense(7), name='proj', input='input')
    graph.add_node(LSTM(7, return_sequences=True), name='lstm', input='proj')
    graph.add_node(Lambda(custom_fun, output_shape=[10, 7]), merge_mode = 'join', name = 'merge', inputs=['proj','lstm'])
    graph.add_node(TimeDistributedDense(39), name='proj2', input='merge')
    graph.add_output(name='output', input='proj2')

    X_train = np.random.rand(64, 10, 39)
    graph.compile(optimizer='rmsprop', loss={'output':'mse'})
    history = graph.fit({'input':X_train, 'output':X_train}, nb_epoch=10)



def speed_limit(): 
    #almost there omg
    def custom_fun(X):
        from keras.objectives import mean_squared_error as mse
        print X.keys()
        return mse(X['proj'], X['lstm'])

    graph = Graph()
    graph.add_input(name='input', input_shape=[10, 39])
    graph.add_node(TimeDistributedDense(7), name='proj', input='input')

    graph.add_node(LSTM(7, return_sequences=True), name='lstm', input='proj')
    graph.add_node(Lambda(custom_fun, output_shape=[10, 7]), merge_mode = 'join', name = 'diff', inputs=['proj','lstm'])
    graph.add_output(name = 'reconstruction', input='diff')

    graph.add_node(TimeDistributedDense(39), name='proj2', input='lstm')
    graph.add_output(name='output', input='proj2')

    speed = 0
    X_train = np.random.rand(64, 10, 39)
    zero_vec = np.ones([64, 10]) *speed
    graph.compile(optimizer='rmsprop', loss={'output':'mse', 'reconstruction':'mse'})
    history = graph.fit({'input':X_train, 'output':X_train, 'reconstruction': zero_vec}, nb_epoch=10)




def custom_fun(X):
    from keras.objectives import mean_squared_error as mse
    print X.keys()
    return mse(X['proj'], X['lstm'])

graph = Graph()
graph.add_input(name='input', input_shape=[10, 39])
graph.add_node(TimeDistributedDense(7), name='proj', input='input')

graph.add_node(LSTM(7, return_sequences=True), name='lstm', input='proj')
graph.add_node(Lambda(custom_fun, output_shape=[10, 7]), merge_mode = 'join', name = 'diff', inputs=['proj','lstm'])
graph.add_output(name = 'reconstruction', input='diff')

graph.add_node(TimeDistributedDense(39), name='proj2', input='lstm')
graph.add_output(name='output', input='proj2')

speed = 0
X_train = np.random.rand(64, 10, 39)
zero_vec = np.ones([64, 10]) *speed
graph.compile(optimizer='rmsprop', loss={'output':'mse', 'reconstruction':'mse'})
history = graph.fit({'input':X_train, 'output':X_train, 'reconstruction': zero_vec}, nb_epoch=10)
