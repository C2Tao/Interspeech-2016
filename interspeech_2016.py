from parser import get_X, get_y
from highway import *

ark_path = 'global_phone_feats/train.39.cmvn.ark'
label_path = 'global_phone_feats/train_query1_label'
X_train, utt_list, max_dur, normalizer = get_X(ark_path)
y_train, utt_list =  get_y(label_path)



n_raw_dim = 39
n_max_dur = 10
n_feat_dim = 7
n_batch_size = 64

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
