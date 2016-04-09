import cPickle
from keras.layers import containers, Dropout, LSTM
import theano
import theano.tensor as T
from keras.optimizers import RMSprop
from keras import models
from keras.utils import generic_utils
import numpy as np
from keras.layers.core import  Activation, AutoEncoder, Dense, TimeDistributedDense, TimeDistributedMerge, Merge, Lambda, MaxoutDense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from recurrent import Highway_LSTM, Highway_GRU, Highway_SimpleRNN
from keras.models import Sequential, Graph, model_from_yaml
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
import keras.backend as K




class GateDense(Dense):
    def build(self):
        input_dim = self.input_shape[1]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.b = K.ones((self.output_dim,), name='{}_b'.format(self.name))
        #self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

        self.trainable_weights = [self.W, self.b]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


class GateTimeDistributedDense(TimeDistributedDense):
    def build(self):
        input_dim = self.input_shape[2]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.b = K.ones((self.output_dim,), name='{}_b'.format(self.name))
        #self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

def fun_speed(inputs):
    from keras.objectives import mean_squared_error as mse
    k0, k1 = inputs.keys()
    return mse(inputs[k0], inputs[k1])

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def fun_speed_dnn(inputs):
    from keras.objectives import mean_squared_error as mse
    k0, k1 = inputs.keys()
    return K.mean(K.square(inputs[k0] - inputs[k1]), axis=-1)

def fun_highway(inputs):
    for k in inputs.keys():
        if 'orig' in k: f = inputs[k]
        elif 'gate' in k: w = inputs[k]
        else: x = inputs[k]
    return w * x + (1.0 - w) * f
    
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


class RestrictedDNN(object):
    def __init__(self, dnn_type, connection, dimensions):
            
        self.dnn_type = dnn_type
        self.connection = connection
        
        self.nF = dimensions['nF'] #39#39
        self.nH = dimensions['nH'] #7#100
        
        self.restriction = []
        self.speed_limit = {}
        self.speed_fine = {}

        self.dnn_layer = []
        self.layer_type = {}
        self.layer_con = {}

        self.graph = Graph()
        
    def get_dnn(self, dnn_type = None):
        if not dnn_type: dnn_type = self.dnn_type
        if dnn_type == 'Maxout':
            return MaxoutDense
        elif dnn_type == 'Dense':
            return Dense

    def add_speed_limit(self, input_name, layer_name, speed_limit = 0.0, speed_fine = 1.0):
        # force outputs of input_name and layer_name to be similar
        pair_name = '|'+'-'.join(sorted([input_name, layer_name]))+'|'
        self.graph.add_node(\
            Lambda(fun_speed_dnn, output_shape = [1]), \
            merge_mode = 'join', name = 'diff_' + pair_name, \
            inputs = [input_name, layer_name])
        print 'sp in ',self.graph.nodes['diff_'+pair_name].input_shape
        print 'sp out',self.graph.nodes['diff_'+pair_name].output_shape
        self.graph.add_output(name = pair_name, input='diff_' + pair_name)
        self.restriction.append(pair_name)
        self.speed_limit[pair_name] = speed_limit
        self.speed_fine[pair_name] = speed_fine

    def add_input(self, layer_name = None):
        if not layer_name:
            layer_name = '_'.join([str(len(self.dnn_layer)), 'dense', 'input'])

        self.graph.add_input(name='input', input_shape=[self.nF])
        self.graph.add_node(Dense(self.nH), name=layer_name, input='input')
        
        self.dnn_layer.append(layer_name)
        self.layer_type[layer_name] = 'dense'
        self.layer_con[layer_name] = 'input'

    def add_output(self, layer_name = None, dnn_type = None):
        if not dnn_type: 
            dnn_type = self.dnn_type
        if not layer_name:
            layer_name = '_'.join([str(len(self.dnn_layer)), dnn_type, 'output'])
        dnn = self.get_dnn(dnn_type)
        input_name = self.dnn_layer[-1] 
        
        self.graph.add_node(dnn(self.nH), name = layer_name, input = input_name)
        self.graph.add_node(Dense(1, activation = 'sigmoid'), name = 'final', input = layer_name)
        print self.graph.nodes['final'].input_shape
        print self.graph.nodes['final'].output_shape
        self.graph.add_output(name='output', input='final')
        
        self.dnn_layer.append(layer_name)
        self.layer_type[layer_name] = self.dnn_type
        self.layer_con[layer_name] = 'output'

    def add_dnn(self, layer_name = None, dnn_type = None, connection = None):
        if not connection: 
            connection = self.connection
        if not dnn_type: 
            dnn_type = self.dnn_type
        if not layer_name:
            layer_name = '_'.join([str(len(self.dnn_layer)), dnn_type, connection])
        dnn = self.get_dnn(dnn_type)
        input_name = self.dnn_layer[-1] 
       
        if connection == 'residual':
            self.add_residual(input_name, layer_name, dnn)
        elif connection == 'vanilla':
            self.add_vanilla(input_name, layer_name, dnn)
        if connection == 'highway':
            self.add_highway(input_name, layer_name, dnn)
            
        
        self.dnn_layer.append(layer_name)
        self.layer_type[layer_name] = dnn_type
        self.layer_con[layer_name] = connection
         
        
    def add_highway(self,  input_name, layer_name, dnn):
        # x: input_name
        # f(x): dnn_layer_name
        # w(x): gate_layer_name
        # y(x): layer_name
        # y(x) = w(x)*f(x) + (1-w(x))*x
        print  self.graph.nodes[input_name].output_shape 
        self.graph.add_node(dnn(self.nH), name = 'orig_' + layer_name, input = input_name)
        self.graph.add_node(GateDense(self.nH, activation='sigmoid'), name = 'gate_' + layer_name, input = input_name)
        
            #Lambda(fun_highway, output_shape = self.graph.nodes[input_name].output_shape), \
        
        self.graph.add_node(\
            Lambda(fun_highway, output_shape = [self.nH]), \
            merge_mode = 'join', name = layer_name, \
            inputs = [input_name, 'orig_' + layer_name, 'gate_' + layer_name])

    def add_residual(self, input_name, layer_name, dnn):
        # x: input_name
        # f(x): dnn_layer_name
        # y(x): layer_name
        # y(x) = f(x) + x
        print  self.graph.nodes[input_name].output_shape 
        self.graph.add_node(dnn(self.nH), name = 'orig_' + layer_name, input = input_name)
        
        self.graph.add_node(\
            Lambda(fun_residual, [self.nH]), \
            merge_mode = 'join', name = layer_name, \
            inputs = [input_name, 'orig_' + layer_name])

    def add_vanilla(self, input_name, layer_name, dnn):
        # x: input_name
        # f(x): layer_name
        # y(x) = f(x)
        print  self.graph.nodes[input_name].output_shape 
        self.graph.add_node(dnn(self.nH), name = layer_name, input = input_name)
    
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
        one_vec = np.ones([nX])
        for layer_name in self.restriction:
            xyio_dict[layer_name] = one_vec * self.speed_limit[layer_name]
        for k in xyio_dict:
            print k, xyio_dict[k].shape
        return xyio_dict
    
    def compile(self):
        self.get_loss()
        self.get_loss_weights()
        
        self.graph.compile(optimizer='adam', loss = self.loss, loss_weights = self.loss_weights)
        #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        #self.graph.compile(optimizer=sgd, loss = self.loss, loss_weights = self.loss_weights)
        

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
        print '    ', self.graph.inputs.keys()

        print "graph dnn_nodes:"
        #print [k for k in self.graph.nodes.keys() if not('diff' in k or 'orig' in k or '|' in k or 'gate' in k)]
        for k in self.dnn_layer:
            print '    ', k, self.layer_type[k], self.layer_con[k]
        
        print "graph speed_limits:"
        print '    ', [k for k in self.graph.nodes.keys() if 'diff' in k]
        
        print "graph outputs:"
        print '    ', self.graph.outputs.keys()

        print "objecives error:"
        print '    ', self.loss

        print "objecives weights:"
        print '    ', self.loss_weights
        print "=========================================="
        if info_path:
            sys.stdout = sys_out

    def fit(self, train_Xy, val_Xy, nb_epoch = 100, patience = 5, model_path='model/tmp'):
        X, y = train_Xy
        vX, vy = val_Xy

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, verbose=1)

        xyio_dict = self.get_xyio({'input':X, 'output':y}, X.shape[0])
        xyval_dict = self.get_xyio({'input':vX, 'output':vy}, vX.shape[0])
        self.history = self.graph.fit(xyio_dict, validation_data = xyval_dict, nb_epoch=nb_epoch, callbacks=[model_checkpoint, early_stopping])
    
    def save(self, model_path):
        self.print_graph(model_path+'.txt')
        self.graph.save_weights(model_path)
        
    def load(self, model_path):
        #with open(model_path+'.txt','r') as f:
        #    for line in f: print line.strip()
        self.graph.load_weights(model_path) 



if __name__=='__main__':
    nX = 10
    nV = 2
    dim = {'nF': 3, 'nH': 7}
    X_train = np.random.rand(nX, dim['nF'])
    y_train = np.random.randint(2, size = (nX,) )
    
    X_val = np.random.rand(nV, dim['nF'])
    y_val = np.random.randint(2, size = (nV,) )
    
    rdnn = RestrictedDNN('Dense', 'highway', dim) 
    
    rdnn.add_input('proj')
    rdnn.add_dnn('dnn1', 'Dense', 'highway')
    rdnn.add_dnn('dnn2', 'Dense', 'residual')
    rdnn.add_dnn('dnn3', 'Dense', 'vanilla')
    rdnn.add_dnn('dnn4')
    rdnn.add_output() 
    
    rdnn.add_speed_limit('proj', 'dnn4', speed_limit = 0, speed_fine =1)
    #rdnn.add_speed_limit('dnn1', 'dnn2', speed_fine = -1)
    #rdnn.add_speed_limit('dnn2', 'dnn3' )
    
    rdnn.compile()
    rdnn.fit((X_train, y_train),(X_val, y_val), 100, 5)

    rdnn.save('test') 
    rdnn.load('test') 

    print rdnn.evaluate(X_train)



