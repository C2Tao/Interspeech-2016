import numpy as np

class Ark(object):
    def __init__(self, ark_path, max_dur = None, normalizer = None):
        self.ark_path = ark_path
        
        #parse feature
        self.feat = {}
        self._parse_ark()
        
        #get dimensions
        self.nF = 0 # feat_dim
        self.nU = 0 # corpus_size
        self._init_dims()

        #transform feat to 3D numpy array
        self.X = None
        self.nT = 0
        self._pad_feats(max_dur)
        
        #normalize feat
        self.normalizer = normalizer
        self._normalize(normalizer)

    def _parse_ark(self):
        # returns dict[utt_id] = np.array[#duration x #feat_dim]
        feat = {}
        with open(self.ark_path) as f:
            for line in f:    
                if '[' in line:
                    utt =  line.split('[')[0].strip()
                    feat[utt] = []
                elif ']' in line:
                    feat[utt].append(map(float,line.strip().split()[:-1]))
                    feat[utt] = np.array(feat[utt],dtype=np.float32)
                else:
                    feat[utt].append(map(float,line.strip().split()))
        self.feat = feat
        self.utt_list = sorted(feat.keys())

    def _init_dims(self):
        self.nU = len(self.utt_list)
        self.nF = self.feat.values()[0].shape[1]
        self._dur_max = 0
        dur_list = []
        for utt in self.utt_list:
            dur = self.feat[utt].shape[0]
            if self._dur_max < dur:
                self._dur_max = dur
            dur_list.append(dur)
        self._dur_mean = np.mean(dur_list)
        self._dur_std = np.std(dur_list)
        self._dur_95 = int(self._dur_mean + self._dur_std*2)

    def _pad_feats(self, max_dur = None):
        if not max_dur:
            max_dur = self._dur_95
            print "padding feats to 95 percentile len:", max_dur
        self.nT = max_dur
        X = np.zeros((self.nU, self.nT, self.nF),dtype=np.float32)
        for i, utt in enumerate(self.utt_list):
            nT, _ = self.feat[utt].shape
            b = max(0, self.nT - nT)
            e = min(self.nT, nT)
            X[i, b:, :] = self.feat[utt][:e, :]
        self.X = X

    def _normalize(self, normalizer = None):
        X = self.X.reshape(self.nU*self.nT, self.nF)
        if not normalizer:
            from sklearn.preprocessing import StandardScaler
            print "calculating gaussian stats on current dataset with shape:", self.X.shape
            normalizer = StandardScaler() 
            normalizer.fit(X)
        self.X = normalizer.transform(X).reshape(self.nU, self.nT, self.nF)
        self.normalizer = normalizer

class Label(object):
    def __init__(self, label_path):
        self.label_path = label_path

        #parse label 
        self.label = {}
        self._parse_label()
        self.utt_list = sorted(self.label.keys()) 
        self.nU = len(self.utt_list)
        
        #get y
        self.y = []
        self._sort_y()
        
    def _sort_y(self):
        self.y = []
        for i, utt in enumerate(sorted(self.utt_list)):
            self.y.append(self.label[utt])
        self.y = np.array(self.y, dtype = np.int)

    
    def _parse_label(self):
        # returns dict[utt_id] = np.array[#duration x #nF]
        self.label = {}
        with open(self.label_path) as f:
            for line in f:   
                utt, y = line.strip().split() 
                self.label[utt] = int(y)
        
if __name__=='__main__':
    ark_path = 'global_phone_feat/Czech_wrd_feats/train.39.cmvn.ark'
    #label_path = 'global_phone_feat/train_query1_label'
    label_path = 'global_phone_std/Czech/selected/wrd_corpus/train/train_Czech_query0_label'

    ark  = Ark(ark_path)
    label  = Label(label_path)
    
    
