# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.layers.core import MaskedLayer
from keras.layers.recurrent import Recurrent, time_distributed_dense

class Highway_LSTM(Recurrent):
    '''Long-Short Term Memory unit - Hochreiter 1997.

    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.highway_bias_init = initializations.get('one')
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        super(Highway_LSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))
        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))
        ####h for highway 
        self.W_w = self.init((input_dim, self.output_dim),
                             name='{}_W_w'.format(self.name))
        self.U_w = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_w'.format(self.name))
        self.b_w = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_w'.format(self.name))
        ###
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_c,
                                                        self.W_o,
                                                        self.W_w]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                        self.U_f,
                                                        self.U_c,
                                                        self.U_o,
                                                        self.U_w]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                        self.b_f,
                                                        self.b_c,
                                                        self.b_o,
                                                        self.b_w]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o,
                                  self.W_w, self.U_w, self.b_w]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x, train=False):
        if train and (0 < self.dropout_W < 1):
            dropout = self.dropout_W
        else:
            dropout = 0
        input_shape = self.input_shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]

        x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_w = time_distributed_dense(x, self.W_w, self.b_w, dropout,
                                     input_dim, self.output_dim, timesteps)
        #return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        return K.concatenate([x_i, x_f, x_c, x_o, x_w, x], axis=2)

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        if len(states) == 3:
            B_U = states[2]
        else:
            B_U = [1. for _ in range(6)]#
        x_i = x[:, :self.output_dim]
        x_f = x[:, self.output_dim: 2 * self.output_dim]
        x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
        x_o = x[:, 3 * self.output_dim: 4 * self.output_dim]
        x_w = x[:, 4 * self.output_dim: 5 * self.output_dim]#
        x_0 = x[:, 5 * self.output_dim:]#

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))
        w = self.inner_activation(x_w + K.dot(h_tm1 * B_U[4], self.U_w))#
        h = w * x_0 + (1.0 - w) * o * self.activation(c)
        #h = o * self.activation(c)
        


        return h, [h, c]

    def get_constants(self, x, train=False):
        if train and (0 < self.dropout_U < 1):
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(6)]
            return [B_U]
        return []

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(Highway_LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Highway_GRU(Recurrent):
    '''Gated Recurrent Unit - Cho et al. 2014.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.highway_bias_init = initializations.get('one')
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        super(Highway_GRU, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W_z = self.init((input_dim, self.output_dim),
                             name='{}_W_z'.format(self.name))
        self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_z'.format(self.name))
        self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

        self.W_r = self.init((input_dim, self.output_dim),
                             name='{}_W_r'.format(self.name))
        self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_r'.format(self.name))
        self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

        self.W_h = self.init((input_dim, self.output_dim),
                             name='{}_W_h'.format(self.name))
        self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_h'.format(self.name))
        self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))
        
        self.W_w = self.init((input_dim, self.output_dim),
                             name='{}_W_w'.format(self.name))
        self.U_w = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_w'.format(self.name))
        self.b_w = self.highway_bias_init((self.output_dim,),
                                         name='{}_b_w'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_z,
                                                        self.W_r,
                                                        self.W_h, 
                                                        self.W_w]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_z,
                                                        self.U_r,
                                                        self.U_h,
                                                        self.U_w]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_z,
                                                        self.b_r,
                                                        self.b_h,
                                                        self.b_w]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                  self.W_r, self.U_r, self.b_r,
                                  self.W_h, self.U_h, self.b_h,
                                  self.W_w, self.U_w, self.b_w]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x, train=False):
        if train and (0 < self.dropout_W < 1):
            dropout = self.dropout_W
        else:
            dropout = 0
        input_shape = self.input_shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]

        x_z = time_distributed_dense(x, self.W_z, self.b_z, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_r = time_distributed_dense(x, self.W_r, self.b_r, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_h = time_distributed_dense(x, self.W_h, self.b_h, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_w = time_distributed_dense(x, self.W_w, self.b_w, dropout,
                                     input_dim, self.output_dim, timesteps)
        return K.concatenate([x_z, x_r, x_h, x_w, x], axis=2)

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        if len(states) == 2:
            B_U = states[1]  # dropout matrices for recurrent units
        else:
            B_U = [1., 1., 1., 1., 1.]

        x_z = x[:, :self.output_dim]
        x_r = x[:, self.output_dim: 2 * self.output_dim]
        x_h = x[:, 2 * self.output_dim: 3 * self.output_dim]
        x_w = x[:, 3 * self.output_dim: 4 * self.output_dim]
        x_0 = x[:, 4 * self.output_dim:]

        z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
        r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))
        w = self.inner_activation(x_w + K.dot(h_tm1 * B_U[3], self.U_w))

        hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = w * x_0 + (1.0 - w) * (z * h_tm1 + (1 - z) * hh)
        return h, [h]

    def get_constants(self, x, train=False):
        if train and (0 < self.dropout_U < 1):
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(5)]
            return [B_U]
        return []

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(Highway_GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Highway_SimpleRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get('hard_sigmoid')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        super(Highway_SimpleRNN, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
        
        self.W_w = self.init((input_dim, self.output_dim),
                           name='{}_W_w'.format(self.name))
        self.U_w = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U_w'.format(self.name))
        self.b_w = K.ones((self.output_dim,), name='{}_b_w'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.W_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.W_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x, train=False):
        if train and (0 < self.dropout_W < 1):
            dropout = self.dropout_W
        else:
            dropout = 0
        input_shape = self.input_shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]
        x_h = time_distributed_dense(x, self.W, self.b, dropout,
                                      input_dim, self.output_dim, timesteps)
        x_w = time_distributed_dense(x, self.W_w, self.b_w, dropout,
                                      input_dim, self.output_dim, timesteps)
        return K.concatenate([x_h, x_w, x], axis=2)

    def step(self, x, states):
        prev_output = states[0]
        if len(states) == 2:
            B_U = states[1]
        else:
            B_U = [1., 1., 1.]
        
        x_h = x[:, :self.output_dim]
        x_w = x[:, self.output_dim: 2 * self.output_dim]
        x_0 = x[:, 2 * self.output_dim:]
        
        w = self.inner_activation(x_h + K.dot(prev_output * B_U[1], self.U_w))
        output = w * x_0  + (1.0 - w) * self.activation(x_h + K.dot(prev_output * B_U[0], self.U))
        # still needs work
        return output, [output]

    def get_constants(self, x, train=False):
        if train and (0 < self.dropout_U < 1):
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = K.dropout(ones, self.dropout_U)
            return [B_U]
        return []

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(Highway_SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

