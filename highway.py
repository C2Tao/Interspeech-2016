import cPickle
from keras.layers import containers, Dropout, LSTM
import theano
import theano.tensor as T
from keras.optimizers import RMSprop
from keras import models
from keras.utils import generic_utils
import numpy as np
from keras.layers.core import  Activation, AutoEncoder, Dense, TimeDistributedDense, TimeDistributedMerge, Merge, Lambda
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential, Graph, model_from_yaml 
import sys

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


class Logger(object):
    def __init__(self, info_path):
        self.terminal = sys.stdout
        self.log = open(info_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

class RestrictedRNN(object):
    def __init__(self, rnn_type, connection, dimensions):
            
        self.rnn_type = rnn_type
        self.connection = connection
        
        self.nT = dimensions['nT'] #10#400
        self.nF = dimensions['nF'] #39#39
        self.nH = dimensions['nH'] #7#100
        
        self.restriction = []
        self.speed_limit = {}
        self.speed_fine = {}

        self.rnn_layer = []
        self.layer_type = {}
        self.layer_con = {}

        self.graph = Graph()
        
    def get_rnn(self, rnn_type = None):
        if not rnn_type: rnn_type = self.rnn_type
        if rnn_type == 'LSTM':
            return LSTM
        elif rnn_type == 'GRU':
            return GRU
        elif rnn_type == 'SimpleRNN':
            return SimpleRNN

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

    def add_input(self, layer_name = None):
        if not layer_name:
            layer_name = '_'.join([str(len(self.rnn_layer)), 'dense', 'input'])

        self.graph.add_input(name='input', input_shape=[self.nT, self.nF])
        self.graph.add_node(TimeDistributedDense(self.nH), name=layer_name, input='input')
        
        self.rnn_layer.append(layer_name)
        self.layer_type[layer_name] = 'dense'
        self.layer_con[layer_name] = 'input'

    def add_output(self, layer_name = None, rnn_type = None):
        if not rnn_type: 
            rnn_type = self.rnn_type
        if not layer_name:
            layer_name = '_'.join([str(len(self.rnn_layer)), rnn_type, 'output'])
        rnn = self.get_rnn(rnn_type)
        input_name = self.rnn_layer[-1] 
        
        self.graph.add_node(rnn(self.nH, return_sequences=False), name = layer_name, input = input_name)
        self.graph.add_node(Dense(1, activation = 'sigmoid'), name = 'final', input = layer_name)
        self.graph.add_output(name='output', input='final')
        
        self.rnn_layer.append(layer_name)
        self.layer_type[layer_name] = self.rnn_type
        self.layer_con[layer_name] = 'none'

    def add_rnn(self, layer_name = None, rnn_type = None, connection = None):
        if not connection: 
            connection = self.connection
        if not rnn_type: 
            rnn_type = self.rnn_type
        if not layer_name:
            layer_name = '_'.join([str(len(self.rnn_layer)), rnn_type, connection])
        rnn = self.get_rnn(rnn_type)
        input_name = self.rnn_layer[-1] 
       
        if connection == 'highway':
            self.add_highway(input_name, layer_name, rnn)
        elif connection == 'residual':
            self.add_residual(input_name, layer_name, rnn)
        elif connection == 'vanilla':
            self.add_vanilla(input_name, layer_name, rnn)
        
        self.rnn_layer.append(layer_name)
        self.layer_type[layer_name] = rnn_type
        self.layer_con[layer_name] = connection
         
        
    def add_highway(self,  input_name, layer_name, rnn):
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

    def add_residual(self, input_name, layer_name, rnn):
        # x: input_name
        # f(x): rnn_layer_name
        # y(x): layer_name
        # y(x) = f(x) + x
        self.graph.add_node(rnn(self.nH, return_sequences=True), name = 'orig_' + layer_name, input = input_name)
        self.graph.add_node(\
            Lambda(fun_residual, output_shape = [self.nT, self.nH]), \
            merge_mode = 'join', name = layer_name, \
            inputs = [input_name, 'orig_' + layer_name])

    def add_vanilla(self, input_name, layer_name, rnn):
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
        

        t_in = self.graph.inputs['input'].get_input()
        t_out = self.graph.outputs['output'].get_output()
        self.evaluate = theano.function([t_in], t_out, allow_input_downcast=True)
        
        self.print_graph()

    
    def print_graph(self, info_path=None):
        if info_path:
            sys_out = sys.stdout
            sys.stdout = Logger(info_path)
        print "=========================================="
        print "graph inputs:"
        print self.graph.inputs.keys()

        print "graph rnn_nodes:"
        #print [k for k in self.graph.nodes.keys() if not('diff' in k or 'orig' in k or '|' in k or 'gate' in k)]
        for k in self.rnn_layer:
            print k, self.layer_type[k], self.layer_con[k]
        
        print "graph speed_limits:"
        print [k for k in self.graph.nodes.keys() if 'diff' in k]
        
        print "graph outputs:"
        print self.graph.outputs.keys()

        print "objecives error:"
        print self.loss

        print "objecives weights:"
        print self.loss_weights
        print "=========================================="
        if info_path:
            sys.stdout = sys_out

    def fit(self, train_Xy, val_Xy, nb_epoch = 100, patience = 5):
        X, y = train_Xy
        vX, vy = val_Xy

        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        xyio_dict = self.get_xyio({'input':X, 'output':y}, X.shape[0])
        xyval_dict = self.get_xyio({'input':vX, 'output':vy}, vX.shape[0])
        self.history = self.graph.fit(xyio_dict, validation_data = xyval_dict, nb_epoch=nb_epoch, callbacks=[early_stopping])
    
    def save(self, model_path):
        self.print_graph(model_path+'.txt')
        self.graph.save_weights(model_path)
        
    def load(self, model_path):
        with open(model_path+'.txt','r') as f:
            for line in f: print line.strip()
        self.graph.load_weights(model_path) 


if __name__=='__main__':
    nX = 10
    nV = 2
    dim = {'nT': 7, 'nF': 3, 'nH': 7}
    X_train = np.random.rand(nX, dim['nT'], dim['nF'])
    y_train = np.random.randint(2, size = (nX,) )
    
    X_val = np.random.rand(nV, dim['nT'], dim['nF'])
    y_val = np.random.randint(2, size = (nV,) )
    
    rrnn = RestrictedRNN('LSTM', 'highway', dim) 
    
    rrnn.add_input('proj')
    rrnn.add_rnn('lstm1', 'LSTM', 'highway')
    rrnn.add_rnn('lstm2', 'SimpleRNN', 'residual')
    rrnn.add_rnn('lstm3', 'GRU', 'vanilla')
    rrnn.add_rnn('lstm4')
    rrnn.add_output() 
    
    rrnn.add_speed_limit('proj', 'lstm4', speed_limit = 0, speed_fine =1)
    rrnn.add_speed_limit('lstm1', 'lstm2', speed_fine = -1)
    rrnn.add_speed_limit('lstm2', 'lstm3' )
    
    rrnn.compile()
    rrnn.fit((X_train, y_train),(X_val, y_val), 100, 5)

    rrnn.save('test') 
    rrnn.load('test') 

    print rrnn.evaluate(X_train)



