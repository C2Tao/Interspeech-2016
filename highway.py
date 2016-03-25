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


def fun_speed(inputs):
    from keras.objectives import mean_squared_error as mse
    k0, k1 = inputs.keys()
    return mse(inputs[k0], inputs[k1])

def fun_highway(inputs):
    for k in inputs.keys():
        if 'orig' in k: f = inputs[k]
        elif 'gate' in k: w = inputs[k]
        else: x = inputs[k]
    return w * f + (1.0 - w) * x
    
def fun_residual(inputs):
    k0, k1 = inputs.keys()
    return inputs[k0] + inputs[k1]


class RestrictedRNN(object):
    def __init__(self, rnn, connection, dimensions):
        self.rnn = rnn
        self.connection = connection
        
        self.nT = dimensions['nT'] #10#400
        self.nF = dimensions['nF'] #39#39
        self.nH = dimensions['nH'] #7#100
        
        self.restriction = []
        self.speed_limit = {}
        self.speed_fine = {}
        self.graph = Graph()


    def add_speed_limit(self, input_name, layer_name, speed_limit = 0.0, speed_fine = 1.0):
        # force outputs of input_name and layer_name to be similar
        pair_name = '|'+'-'.join(sorted([input_name, layer_name]))+'|'
        self.graph.add_node(\
            Lambda(fun_speed, output_shape = [self.nT, self.nH]),\
            merge_mode = 'join', name = 'diff_' + pair_name, \
            inputs = [input_name, layer_name])
        self.graph.add_output(name = pair_name, input='diff_' + pair_name)
        self.restriction.append(pair_name)
        self.speed_limit[pair_name] = speed_limit
        self.speed_fine[pair_name] = speed_fine

    def add_input(self, layer_name):
        self.graph.add_input(name='input', input_shape=[self.nT, self.nF])
        self.graph.add_node(TimeDistributedDense(self.nH), name=layer_name, input='input')

    def add_output(self, layer_name): 
        self.graph.add_node(self.rnn(self.nH, return_sequences=False), name = 'top', input = layer_name)
        self.graph.add_node(Dense(1, activation = 'sigmoid'), name = 'final', input = 'top')
        self.graph.add_output(name='output', input='final')

    def add_rnn(self, input_name, layer_name, rnn = None, connection = None):
        if not rnn: rnn = self.rnn
        if not connection: connection = self.connection

        if connection == 'highway':
            self.add_highway(input_name, layer_name, rnn)
        elif connection == 'residual':
            self.add_residual(input_name, layer_name, rnn)
        elif connection == 'vanilla':
            self.add_vanilla(input_name, layer_name, rnn)
        
    def add_highway(self,  input_name, layer_name, rnn = None):
        if not rnn: rnn = self.rnn
        # x: input_name
        # f(x): rnn_layer_name
        # w(x): gate_layer_name
        # y(x): layer_name
        # y(x) = w(x)*f(x) + (1-w(x))*x
        self.graph.add_node(rnn(self.nH, return_sequences=True), name = 'orig_' + layer_name, input = input_name)
        self.graph.add_node(TimeDistributedDense(self.nH, activation='sigmoid'), name = 'gate_' + layer_name, input = input_name)
        self.graph.add_node(\
            Lambda(fun_highway, output_shape = [self.nT, self.nH]), \
            merge_mode = 'join', name = layer_name, \
            inputs = [input_name, 'orig_' + layer_name, 'gate_' + layer_name])

    def add_residual(self, input_name, layer_name, rnn = None):
        if not rnn: rnn = self.rnn
        # x: input_name
        # f(x): rnn_layer_name
        # y(x): layer_name
        # y(x) = f(x) + x
        self.graph.add_node(rnn(self.nH, return_sequences=True), name = 'orig_' + layer_name, input = input_name)
        self.graph.add_node(\
            Lambda(fun_residual, output_shape = [self.nT, self.nH]), \
            merge_mode = 'join', name = layer_name, \
            inputs = [input_name, 'orig_' + layer_name])

    def add_vanilla(self, input_name, layer_name, rnn = None):
        if not rnn: rnn = self.rnn
        # x: input_name
        # f(x): layer_name
        # y(x) = f(x)
        self.graph.add_node(rnn(self.nH, return_sequences=True), name = layer_name, input = input_name)
    
    def get_loss(self):
        loss_dict = {'output': 'binary_crossentropy'}
        for layer_name in self.restriction:
            loss_dict[layer_name] = 'mse'
        self.loss = loss_dict
        
    def get_loss_weights(self):
        loss_weight_dict = {'output': 1.0}
        for layer_name in self.restriction:
            loss_weight_dict[layer_name] = self.speed_fine[layer_name]
        self.loss_weights = loss_weight_dict
 
    def get_xyio(self, xyio_dict, nX):
        one_vec = np.ones([nX, self.nT])
        for layer_name in self.restriction:
            xyio_dict[layer_name] = one_vec * self.speed_limit[layer_name]
        return xyio_dict
    
    def compile(self):
        self.get_loss()
        self.get_loss_weights()
        self.graph.compile(optimizer='adam', loss = self.loss, loss_weights = self.loss_weights)
        self.print_graph()

    def fit(self, X, y, nb_epoch):
        xyio_dict = self.get_xyio({'input':X, 'output':y}, X.shape[0])
        self.history = self.graph.fit(xyio_dict, nb_epoch=nb_epoch)
    
    def print_graph(self, opt=None):
        print "=========================================="
        print "graph inputs:"
        print self.graph.inputs.keys()

        print "graph rnn_nodes:"
        print [k for k in self.graph.nodes.keys() if not('diff' in k or 'orig' in k or '|' in k or 'gate' in k)]
        
        print "graph speed_limits:"
        print [k for k in self.graph.nodes.keys() if 'diff' in k]
        
        print "graph outputs:"
        print self.graph.outputs.keys()

        print "objecives error:"
        print self.loss

        print "objecives weights:"
        print self.loss_weights
        print "=========================================="
        

if __name__=='__main__':
    nX = 100
    dim = {'nT': 71, 'nF': 39, 'nH': 7}
    X_train = np.random.rand(nX, dim['nT'], dim['nF'])
    y_train = np.random.randint(2, size = (nX,) )
    
    rrnn = RestrictedRNN(LSTM, 'highway', dim) 
    
    rrnn.add_input('proj')
    rrnn.add_rnn('proj', 'lstm1', LSTM, 'highway')
    rrnn.add_rnn('lstm1', 'lstm2', SimpleRNN, 'residual')
    rrnn.add_rnn('lstm2', 'lstm3', GRU, 'vanilla')
    rrnn.add_rnn('lstm3', 'lstm4')
    rrnn.add_output('lstm4') 
    
    rrnn.add_speed_limit('proj', 'lstm4', speed_limit = 0, speed_fine =1)
    rrnn.add_speed_limit('lstm1', 'lstm2', speed_fine = -1)
    rrnn.add_speed_limit('lstm2', 'lstm3' )
    
    rrnn.compile()
    rrnn.fit(X_train, y_train, 10)
    




