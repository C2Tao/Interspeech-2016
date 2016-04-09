from mnist_parse import parse
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = parse()
from highway import HRNN


dim = {'nT': 28, 'nF': 28, 'nH': 28, 'nY': 10}

rrnn = HRNN('GRU', 'residual', dim) 

rrnn.add_input('g0')
rrnn.add_rnn('g1')
rrnn.add_rnn('g2')
rrnn.add_rnn('g3')
rrnn.add_rnn('g4')
rrnn.add_output() 

rrnn.add_speed_limit('input', 'g0', speed_limit = 0, speed_fine = 1)
rrnn.add_speed_limit('g0'   , 'g1'  , speed_limit = 0, speed_fine = 1)
rrnn.add_speed_limit('g1'   , 'g2'  , speed_limit = 0, speed_fine = 1)
rrnn.add_speed_limit('g2'   , 'g3'  , speed_limit = 0, speed_fine = 1)
rrnn.add_speed_limit('g3'   , 'g4'  , speed_limit = 0, speed_fine = 1)

rrnn.compile()
#rrnn.fit((X_train, y_train),(X_valid, y_valid), 100, 5)

#rrnn.save('test') 
rrnn.load('test') 


X = X_train[:10]

vo = rrnn.evaluate(X_test)
v0 = rrnn.visualize['g0'](X)
v1 = rrnn.visualize['g1'](X)
v2 = rrnn.visualize['g2'](X)
v3 = rrnn.visualize['g3'](X)
v4 = rrnn.visualize['g4'](X)
i =7
plt.imshow(np.concatenate([v0[i], v1[i], v2[i], v3[i], v4[i]]))
plt.show()



