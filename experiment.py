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

def model_score(rnn, con, lim, nN, nH, folder = 'score'):
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

def ap_at_n(answer, score, n):
    I = np.array(sorted(range(len(answer)), key=lambda x: score[x], reverse=True))
    sorted_answer = np.array(map(lambda x: float(answer[I[x]]), range(len(answer))))
    position = np.array(range(len(answer))) + 1
    ap = np.cumsum(sorted_answer) / position
    nz = np.nonzero(sorted_answer)[0][:n]
    mapx = np.mean(ap[nz]) 
    return ap[n]

def eval(rnn, con, lim, nN, nH, folder='score'):
    exp_name = '_'.join(map(str, (rnn, con, lim, nH, nN)))
    from sklearn import metrics
    import cPickle

    pred = cPickle.load(open(folder+'/'+exp_name+'.test','r')).flatten()
    y = test_y 
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    from zrst.util import average_precision
    pred = cPickle.load(open(folder+'/'+exp_name+'.test','r')).flatten()
    y = test_y 
    ap = average_precision(y, pred)
    pa5 = ap_at_n(y, pred, 50)
    pa10 = ap_at_n(y, pred, 100)
    #print pa5, pa10, ap, auc
    print ap

#f = model_score

'''
f = eval
nH = 400 
for nL in [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30]:
    for rnn in ['SimpleRNN','LSTM','GRU']:
        if rnn=='SimpleRNN': 
            p = np.sqrt(1.0/2.0)
            q = np.sqrt(1.0/1.0)
        elif rnn=='LSTM':  
            p = np.sqrt(4.0/5.0)
            q = np.sqrt(1.0/4.0)
        elif rnn=='GRU':
            p = np.sqrt(3.0/4.0)
            q = np.sqrt(1.0/3.0)

        print nL, rnn, int(nH*q), int(nH*q*p)
        f(rnn, 'vanilla', None, nL, int(nH*q), 'score_400')
        f(rnn, 'residual', None, nL, int(nH*q), 'score_400')
        f(rnn, 'highway', None, nL, int(nH*q*p), 'score_400')
        f(rnn, 'highway', 0.1, nL, int(nH*q*p), 'score_400')
'''
#f = model_score
f = eval
nH = 100 
for nL in [20, 15, 10, 5]:
    for rnn in ['LSTM','GRU','SimpleRNN']:
        if rnn=='SimpleRNN': 
            p = np.sqrt(1.0/2.0)
            q = np.sqrt(1.0/1.0)
        elif rnn=='LSTM':  
            p = np.sqrt(4.0/5.0)
            q = np.sqrt(1.0/4.0)
        elif rnn=='GRU':
            p = np.sqrt(3.0/4.0)
            q = np.sqrt(1.0/3.0)

        print nL, rnn, int(nH*q), int(nH*q*p)
        f(rnn, 'vanilla', None, nL, int(nH*q))
        f(rnn, 'residual', None, nL, int(nH*q))
        f(rnn, 'highway', None, nL, int(nH*q*p))
        f(rnn, 'highway', 0.1, nL, int(nH*q*p))
        f(rnn, 'highway', 0.01, nL, int(nH*q*p))
        f(rnn, 'highway', 0.001, nL, int(nH*q*p))
