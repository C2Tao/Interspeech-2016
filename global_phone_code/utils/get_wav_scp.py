#From *.utt get the set of utt ID
#and get wav.scp

import sys
loc = sys.argv[1]
language=sys.argv[2]
ori_header=sys.argv[3]
new_header=sys.argv[4]

utt_loc = loc+'/'+language+'.utt'
utt_set = []
with open(utt_loc,'r') as f:
  for line in f.readlines():
    utt = line.rstrip()
    utt_set.append(utt)

ori_scp_loc = new_header+'/GlobalPhone/{}/wav.scp'.format(language)
new_scp_loc = loc+'/{}.wav.scp'.format(language,language)

op = open(new_scp_loc,'w')
with open(ori_scp_loc,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    utt = (line.split())[0]
    if utt not in utt_set:
      continue
    assert(ori_header in line)
    line = line.replace(ori_header,new_header)
    op.write(line+'\n')

op.close()
