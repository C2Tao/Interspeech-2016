#From utt to get .text

import sys
loc = sys.argv[1]
language=sys.argv[2]
corpus_loc=sys.argv[3]

valid_utt = []
op = open('{}/{}.text'.format(loc,language,language),'w')
with open('{}/{}.utt'.format(loc,language,language),'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    utt = (line.split())[0]
    if utt not in valid_utt:
      valid_utt.append(utt)
for target in ['train','dev','test']:
  with open('{}/GlobalPhone/{}/material/{}.word.text'.format(corpus_loc,language,target),'r') as f:
    for line in f.readlines():
      line = line.rstrip()
      utt = (line.split())[0]
      if utt not in valid_utt:
        continue
      else:
        op.write(line+'\n')

  
op.close()


