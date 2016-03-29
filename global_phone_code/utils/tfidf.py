#Assume the tf_corpus is a single line corpus for calculating term grequency
#output: file:chosen_queries which contains the chosen queries
#needs to use get_plainTextSingleLine.py first

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from IPython.core.debugger import Tracer
import sys

loc=sys.argv[1]
language=sys.argv[2]
extract_num = int(sys.argv[3])
start_extract_idx = int(sys.argv[4])

#extract_num = 30
#start_extract_idx = 100

#get multi-pronunciation words
print "  Get multi-pronunciation words"
lexicon_loc = loc+'/lexicon.word.txt'.format(language)
lexicon = []
duplicated_wrd = []
with open(lexicon_loc,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    wrd = (line.split())[0]
    if wrd not in lexicon:
      lexicon.append(wrd)
      continue
    elif wrd in lexicon and wrd not in duplicated_wrd:
      duplicated_wrd.append(wrd)
      continue
#convert the format from string to unicode object
for i in range(0,len(duplicated_wrd)):
  duplicated_wrd[i] = duplicated_wrd[i].decode("utf-8")

#extract text feature
vectorizer = CountVectorizer(min_df=1)
corpus = []
with open('{}/{}_plain.text'.format(loc,language,language),'r') as f:
  for line in f.readlines():
    corpus.append(line.rstrip())

tf_vectorizer = CountVectorizer(min_df=1)
tf_corpus = []
with open('{}/{}_plainForTf.text'.format(loc,language,language),'r') as f:
  for line in f.readlines():
    tf_corpus.append(line.rstrip())

X_train_counts = (vectorizer.fit_transform(corpus)).toarray()

#tf
print "  Calculating tf"
X_tf_counts = ((tf_vectorizer.fit_transform(tf_corpus)).toarray())[0] #assume the tf_corpus is a single line corpus
wrds = tf_vectorizer.get_feature_names()



tf_transformer = TfidfTransformer(use_idf=False).fit(X_tf_counts)
tf = (tf_transformer.transform(X_tf_counts)).toarray()

result = tf  #For tf
wrds = vectorizer.get_feature_names()

high_tf_wrd = []
#sorted_tf_idx = result.argsort()[::-1]
sorted_tf_idx = X_tf_counts.argsort()[::-1] #use counts directly.
for idx in range(start_extract_idx,sorted_tf_idx.shape[0]): #Assume the sorted_tf_idx is a one dimension array
  if wrds[sorted_tf_idx[idx]] in duplicated_wrd:
    print "  Duplicated_pronunciation words! ",wrds[sorted_tf_idx[idx]]
    print "  Skip the the word and find the next one."
  else:
    high_tf_wrd.append(wrds[sorted_tf_idx[idx]])
    if len(high_tf_wrd) >= extract_num:
      print"  Done extraction!"
      break

op = open('{}/chosen_queries'.format(loc,language),'w')
for wrd in high_tf_wrd:
  op.write(wrd.encode('utf-8')+'\n')
op.close()

