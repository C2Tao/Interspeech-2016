from  highway import RestrictedRNN
from parser import Ark, Label
import cPickle
import numpy as np
train_ark_path = 'global_phone_feat/Czech_wrd_feats/train.39.cmvn.ark'
train_label_path = 'global_phone_std/Czech/selected/wrd_corpus/train/train_Czech_query0_label'
test_ark_path = 'global_phone_feat/Czech_wrd_feats/test.39.cmvn.ark'
test_label_path = 'global_phone_std/Czech/selected/wrd_corpus/test/test_Czech_query0_label'
dev_ark_path = 'global_phone_feat/Czech_wrd_feats/dev.39.cmvn.ark'
dev_label_path = 'global_phone_std/Czech/selected/wrd_corpus/dev/dev_Czech_query0_label'
'''
train_ark_path = 'global_phone_feats/train.39.cmvn.ark'
train_label_path = 'global_phone_feats/train_query1_label'
test_ark_path = 'global_phone_feats/test.39.cmvn.ark'
test_label_path = 'global_phone_feats/test_query1_label'
dev_ark_path = 'global_phone_feats/dev.39.cmvn.ark'
dev_label_path = 'global_phone_feats/dev_query1_label'
'''
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
            rrnn.add_speed_limit(rnn+str(i), rnn+str(i+1), speed_limit = 0, speed_fine = 1.0/nN * lim)
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
    rrnn.load('model/'+exp_name+'.model') 
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
    rrnn.load('model/'+exp_name+'.model') 
    dev_p = rrnn.evaluate(dev_X)
    test_p = rrnn.evaluate(test_X)
    train_p = rrnn.evaluate(train_X)
    dump(dev_p, 'score/'+exp_name+'.dev')
    dump(test_p, 'score/'+exp_name+'.test')
    dump(train_p, 'score/'+exp_name+'.train')
    
def eval(rnn, con, lim, nN, nH):
    exp_name = '_'.join(map(str, (rnn, con, lim, nH, nN)))
    from sklearn import metrics
    import cPickle

    pred = cPickle.load(open('score/'+exp_name+'.test','r')).flatten()
    y = test_y 
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print auc

'''
f = model_score
for nL in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]:
    for rnn in ['SimpleRNN','LSTM','GRU']:
        print nL, rnn
        f(rnn, 'vanilla', None, nL, 100)
        f(rnn, 'residual', None, nL, 100)
        f(rnn, 'highway', None, nL, 100)
        f(rnn, 'highway', 1.0, nL, 100)
'''
f = model_score
for nL in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30]:
    for rnn in ['SimpleRNN','LSTM','GRU']:
        print nL, rnn
        f(rnn, 'vanilla', None, nL, 100)
        f(rnn, 'residual', None, nL, 100)
        f(rnn, 'highway', None, nL, 100)
        f(rnn, 'highway', 1.0, nL, 100)
