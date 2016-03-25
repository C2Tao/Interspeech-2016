from  highway import RestrictedRNN
from parser import Ark, Label
import cPickle
train_ark_path = 'global_phone_feats/train.39.cmvn.ark'
train_label_path = 'global_phone_feats/train_query1_label'
test_ark_path = 'global_phone_feats/test.39.cmvn.ark'
test_label_path = 'global_phone_feats/test_query1_label'
dev_ark_path = 'global_phone_feats/dev.39.cmvn.ark'
dev_label_path = 'global_phone_feats/dev_query1_label'

train_ark = Ark(train_ark_path)
train_label  = Label(train_label_path)
train_X, train_y = train_ark.X, train_label.y

test_ark = Ark(test_ark_path, max_dur = train_ark.nT, normalizer = train_ark.normalizer)
test_label  = Label(test_label_path)
test_X, test_y = test_ark.X, test_label.y

dev_ark = Ark(dev_ark_path, max_dur = train_ark.nT, normalizer = train_ark.normalizer)
dev_label  = Label(dev_label_path)
dev_X, dev_y = dev_ark.X, dev_label.y


def dump(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f) 


def compile(rnn, con, lim, nN, nH):
    exp_name = '_'.join(map(str, (rnn, con, lim, nH, nN)))
    dim = {'nT': train_ark.nT, 'nF': train_ark.nF, 'nH': nH}
    
    rrnn = RestrictedRNN(rnn, con, dim)
    rrnn.add_input(rnn+'0')
    for i in range(nN):
        rrnn.add_rnn(rnn+str(i+1), rnn, con)
    rrnn.add_output() 
    if lim:
        for i in range(nN):
            rrnn.add_speed_limit(rnn+str(i), rnn+str(i+1), speed_limit = 0, speed_fine = 1.0/(nN+1.0) * lim)
    rrnn.compile()
    return rrnn

def model(rnn, con, lim, nN, nH):
    exp_name = '_'.join(map(str, (rnn, con, lim, nH, nN)))
    rrnn = compile(rnn, con, lim, nN, nH)
    rrnn.fit((train_X, train_y), (dev_X, dev_y), model_path = 'model/'+exp_name+'.model')
    #rrnn.save('model/'+exp_name) 
    return rrnn

def score(rnn, con, lim, nN, nH, rrnn = None):
    exp_name = '_'.join(map(str, (rnn, con, lim, nH, nN)))
    if not rrnn:
        rrnn = compile(rnn, con, lim, nN, nH)
    rrnn.load('model/'+exp_name) 
    dev_p = rrnn.evaluate(dev_X)
    test_p = rrnn.evaluate(test_X)
    train_p = rrnn.evaluate(train_X)
    dump(dev_p, 'score/'+exp_name+'.dev')
    dump(test_p, 'score/'+exp_name+'.test')
    dump(train_p, 'score/'+exp_name+'.train')
    return rrnn
def model_score(rnn, con, lim, nN, nH):
    exp_name = '_'.join(map(str, (rnn, con, lim, nH, nN)))
    rrnn = compile(rnn, con, lim, nN, nH)
    rrnn.fit((train_X, train_y), (dev_X, dev_y), model_path = 'model/'+exp_name+'.model')
    dev_p = rrnn.evaluate(dev_X)
    test_p = rrnn.evaluate(test_X)
    train_p = rrnn.evaluate(train_X)
    dump(dev_p, 'score/'+exp_name+'.dev')
    dump(test_p, 'score/'+exp_name+'.test')
    dump(train_p, 'score/'+exp_name+'.train')
    


model_score('LSTM', 'vanilla', None, 10, 100)
model_score('LSTM', 'residual', None, 10, 100)
model_score('LSTM', 'highway', None, 10, 100)
model_score('LSTM', 'highway', 1.0, 10, 100)
