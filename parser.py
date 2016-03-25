import os
import numpy as np


def parse_ark(ark_path):
    # returns dict[utt_id] = np.array[#duration x #feat_dim]
    feat = {}
    with open(ark_path) as f:
        for line in f:    
            if '[' in line:
                utt =  line.split('[')[0].strip()
                feat[utt] = []
            elif ']' in line:
                feat[utt].append(map(float,line.strip().split()[:-1]))
                feat[utt] = np.array(feat[utt],dtype=np.float32)
            else:
                feat[utt].append(map(float,line.strip().split()))
    return feat

def parse_label(label_path):
    # returns dict[utt_id] = np.array[#duration x #feat_dim]
    label = {}
    with open(label_path) as f:
        for line in f:   
            utt, y = line.strip().split() 
            label[utt] = int(y)
    return label

def get_dims(feat):
    corpus_size = len(feat.keys())
    max_dur = 0
    feat_dim = feat.values()[0].shape[1]
    dur_list = []
    for utt in sorted(feat.keys()):
        dur = feat[utt].shape[0]
        if max_dur < dur:
            max_dur = dur
        dur_list.append(dur)
    dur_mean = np.mean(dur_list)
    dur_std = np.std(dur_list)
    dur_95 = int(dur_mean + dur_std*2)
    return (corpus_size, max_dur, feat_dim), (dur_mean, dur_std, dur_95)

def pad_feats(feat, max_dur = None):
    # feat: dictionary mapping uttid to numpy array
    # dims: (corpus_size, max_dur, feat_dim), (dur_mean, dur_std, mean+2*std)
    (nU, nT, nF), (du, dv, d95) = get_dims(feat)
    if not max_dur:
        max_dur = d95
        print "padding feats to 95 percentile len:", max_dur
    X = np.zeros((nU, max_dur, nF),dtype=np.float32)
    utt_list = sorted(feat.keys())
    for i, utt in enumerate(utt_list):
        nT, nF = feat[utt].shape
        b = max(0, max_dur - nT)
        e = min(max_dur, nT)
        X[i, b:, :] = feat[utt][:e, :]
    return X, utt_list

def normalize(X, normalizer = None):
    nU, nT, nF = X.shape
    rX = X.reshape(nU*nT, nF)
    if not normalizer:
        from sklearn.preprocessing import StandardScaler
        print "calculating gaussian stats on current dataset with shape:", X.shape
        normalizer = StandardScaler() 
        normalizer.fit(rX)
    nX = normalizer.transform(rX)
    return nX.reshape(nU, nT, nF), normalizer
    
def get_X(ark_path, max_dur = None, normalizer = None):
    feat = parse_ark(ark_path)
    rX, utt_list = pad_feats(feat, max_dur)
    X, normalizer = normalize(rX, normalizer)
    max_len = X.shape
    return X, utt_list, max_dur, normalizer 

def get_y(label_path):
    label = parse_label(label_path)
    y = []
    utt_list = sorted(label.keys()) 
    for i, utt in enumerate(sorted(utt_list)):
        y.append(label[utt])
    y = np.array(y, dtype = np.int)
    return y, utt_list

if __name__=='__main__':
    ark_path = 'global_phone_feats/train.39.cmvn.ark'
    label_path = 'global_phone_feats/train_query1_label'
    '''
    feat = parse_ark(ark_path)
    print parse_label(label_path).items()[:5] 
    (nU, nT, nF), (u, v, d) = get_dims(feat)
    print u, v
    X0,_ = pad_feats(feat, nT)
    X1,_ = pad_feats(feat, 100)
    X2,_ = pad_feats(feat)
    print X0.shape, X1.shape, X2.shape
    normalize(X0)
    x = np.array([ [[1,1],[2,2],[3,3]] , [[4,-4],[5,-5], [6,-6]]  ])    
    y = np.array([ [[1,1],[3,3]] , [[4,-4],[5,-5]]  ])    
    nx,norm = normalize(x)
    print x
    print nx 
    ny,norm = normalize(y,norm)
    print ny
    '''
    X, utt_list_x, max_dur, normalizer = get_X(ark_path)
    y, utt_list_y =  get_y(label_path)
    assert(utt_list_x==utt_list_y)

    
    
