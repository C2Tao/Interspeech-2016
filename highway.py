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



def mse_fun(X):
    from keras.objectives import mean_squared_error as mse
    k0, k1 = X.keys()
    return mse(X[k0], X[k1])


def add_rnn(graph, rnn, input_name, layer_name, n_max_dur, n_feat_dim):
    graph.add_node(rnn(n_feat_dim, return_sequences=True), name=layer_name, input=input_name)
    graph.add_node(Lambda(mse_fun, output_shape=[n_max_dur, n_feat_dim]), merge_mode = 'join', name = 'diff_' + layer_name, inputs = [input_name, layer_name])
    graph.add_output(name = 'speed_' + layer_name, input='diff_' + layer_name)

def add_highway(graph, rnn, input_name):
    graph.add_node(rnn(n_feat_dim, return_sequences=True), name=layer_name, input=input_name)

def add_residual(graph, rnn, input_name):
    pass

n_raw_dim = 39
n_max_dur = 10
n_feat_dim = 7
n_batch_size = 64

graph = Graph()
graph.add_input(name='input', input_shape=[n_max_dur, n_raw_dim])
graph.add_node(TimeDistributedDense(n_feat_dim), name='proj', input='input')

add_rnn(graph, LSTM, 'proj', 'lstm1', n_max_dur, n_feat_dim)
add_rnn(graph, LSTM, 'lstm1', 'lstm2', n_max_dur, n_feat_dim)
add_rnn(graph, LSTM, 'lstm2', 'lstm3', n_max_dur, n_feat_dim)

graph.add_node(TimeDistributedDense(n_raw_dim), name='proj2', input='lstm3')
graph.add_output(name='output', input='proj2')

speed = 0
X_train = np.random.rand(n_batch_size, n_max_dur, n_raw_dim)
zero_vec = np.ones([n_batch_size, n_max_dur]) *speed

loss_dict = {'output':'mse', 'speed_lstm1':'mse', 'speed_lstm2':'mse', 'speed_lstm3':'mse'}
xyio_dict = {'input':X_train, 'output':X_train, 'speed_lstm1': zero_vec, 'speed_lstm2': zero_vec, 'speed_lstm3': zero_vec}

graph.compile(optimizer='adam', loss=loss_dict)
history = graph.fit(xyio_dict, nb_epoch=10)
print graph.nodes.keys()
print graph.outputs.keys()
