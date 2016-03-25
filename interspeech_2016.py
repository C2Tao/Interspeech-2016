from parser import Ark, Label
from highway import *

train_ark_path = 'global_phone_feats/train.39.cmvn.ark'
train_label_path = 'global_phone_feats/train_query1_label'
test_ark_path = 'global_phone_feats/test.39.cmvn.ark'
test_label_path = 'global_phone_feats/test_query1_label'
dev_ark_path = 'global_phone_feats/dev.39.cmvn.ark'
dev_label_path = 'global_phone_feats/dev_query1_label'


train_ark = Ark(train_ark_path)
train_label  = Label(train_label_path)

test_ark = Ark(test_ark_path, max_dur = train_ark.nT, normalizer = train_ark.normalizer)
test_label  = Label(test_label_path)

dev_ark = Ark(dev_ark_path, max_dur = train_ark.nT, normalizer = train_ark.normalizer)
dev_label  = Label(dev_label_path)


n_raw_dim = train_ark.nF
n_max_dur = train_ark.nT
n_feat_dim = 100
n_batch_size = 50
'''
loss = []

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
'''
