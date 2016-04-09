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


n_raw_dim = 39
n_max_dur = 10#10
n_feat_dim = 7#7
n_batch_size = 64#64

loss = []

def get_loss(loss_dict):
    loss_dict
    for layer_name in loss:
        loss_dict['speed_'+layer_name] = 'mse'
    return loss_dict
        
def get_xyio(xyio_dict, speed_limit):
    limit = np.ones([n_batch_size, n_max_dur]) * speed_limit
    for layer_name in loss:
        xyio_dict['speed_'+layer_name] = limit
    return xyio_dict

def fun_speed(inputs):
    from keras.objectives import mean_squared_error as mse
    k0, k1 = inputs.keys()
    return mse(inputs[k0], inputs[k1])

def fun_highway(inputs):
    for k in inputs.keys():
        if 'rnn' in k: f = inputs[k]
        elif 'gate' in k: w = inputs[k]
        else: x = inputs[k]
    return w * f + (1.0 - w) * x
    
def fun_residual(inputs):
    k0, k1 = inputs.keys()
    return inputs[k0] + inputs[k1]



def add_speed_limit(graph, input_name, layer_name):
    # force outputs of input_name and layer_name to be similar
    pair_name = '|'+'-'.join(sorted([input_name, layer_name]))+'|'
        #Lambda(fun_speed, output_shape = [n_max_dur, n_feat_dim]),\
    graph.add_node(\
        Lambda(fun_speed, output_shape = None),\
        merge_mode = 'join', name = 'diff_' + pair_name, \
        inputs = [input_name, layer_name])
    graph.add_output(name = 'speed_' + pair_name, input='diff_' + pair_name)
    loss.append(pair_name)


def add_highway(graph, rnn, input_name, layer_name):
    # x: input_name
    # f(x): rnn_layer_name
    # w(x): gate_layer_name
    # y(x): layer_name
    # y(x) = w(x)*f(x) + (1-w(x))*x
    graph.add_node(rnn(output_dim = n_feat_dim, return_sequences=True), name = 'rnn_' + layer_name, input = input_name)
    graph.add_node(TimeDistributedDense(n_feat_dim, activation='sigmoid'), name = 'gate_' + layer_name, input = input_name)
    graph.add_node(\
        Lambda(fun_highway, output_shape = [n_max_dur, n_feat_dim]), \
        merge_mode = 'join', name = layer_name, \
        inputs = [input_name, 'rnn_' + layer_name, 'gate_' + layer_name])

def add_residual(graph, rnn, input_name, layer_name):
    # x: input_name
    # f(x): rnn_layer_name
    # y(x): layer_name
    # y(x) = f(x) + x
    graph.add_node(rnn(output_dim = n_feat_dim, return_sequences=True), name = 'rnn_' + layer_name, input = input_name)
    graph.add_node(\
        Lambda(fun_residual, output_shape = [n_max_dur, n_feat_dim]), \
        merge_mode = 'join', name = layer_name, \
        inputs = [input_name, 'rnn_' + layer_name])

def add_vanilla(graph, rnn, input_name, layer_name):
    # x: input_name
    # f(x): layer_name
    # y(x) = f(x)
    graph.add_node(rnn(output_dim = n_feat_dim, return_sequences=True), name = layer_name, input = input_name)


graph = Graph()
graph.add_input(name='input', input_shape=[n_max_dur, n_raw_dim])
graph.add_node(TimeDistributedDense(n_feat_dim), name='proj', input='input')

add_highway(graph, LSTM, 'proj', 'lstm')

add_residual(graph, SimpleRNN, 'lstm', 'simple')

add_vanilla(graph, GRU, 'simple', 'gru')

# random constaints
add_speed_limit(graph, 'proj', 'lstm')
add_speed_limit(graph, 'proj', 'gru')
add_speed_limit(graph, 'simple', 'gru')
#



graph.add_node(LSTM(n_feat_dim, return_sequences=False), name = 'top', input = 'gru')

graph.add_node(Dense(1, activation = 'sigmoid'), name = 'final', input = 'top')
graph.add_output(name='output', input='final')

X_train = np.random.rand(n_batch_size, n_max_dur, n_raw_dim)
y_train = np.random.randint(2, size = (n_batch_size,) )

graph.compile(optimizer='adam', loss=get_loss({'output':'binary_crossentropy'}))
history = graph.fit(get_xyio({'input':X_train, 'output':y_train}, 0), nb_epoch=10)


print graph.nodes.keys()
print graph.outputs.keys()
