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

if __name__=='__main__':
    ark_path = 'global_phone_feats/train.39.cmvn.ark'
    label_path = 'global_phone_feats/train_query1_label'
    print parse_ark(ark_path).items()[:2]
    print parse_label(label_path).items()[:5]  
