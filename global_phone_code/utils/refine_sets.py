#re-select utterances containing chosen queries.
import sys
import subprocess
loc = sys.argv[1]
language = sys.argv[2]

subprocess.call('mkdir -p {}/selected'.format(loc),shell=True)

#read chosen queries
choosed_queries = []
chosen_query_text = '{}/chosen_queries'.format(loc)
with open(chosen_query_text,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    choosed_queries.append(line)

ori_text = '{}/{}.text'.format(loc,language)
selected_text = '{}/selected/{}.text'.format(loc,language)
selected_utt = '{}/selected/{}.utt'.format(loc,language)
op_text = open(selected_text,'w')
op_utt = open(selected_utt,'w')

with open(ori_text, 'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    if any(query in line for query in choosed_queries):
      op_text.write(line+'\n')
      op_utt.write((line.split())[0] + '\n')

op_text.close()
op_utt.close()

#re-select wav.scp
choosed_utt = []
with open(selected_utt,'r') as f:
  for line in f.readlines():
    utt = (line.split())[0]
    choosed_utt.append(utt)

selected_wav = '{}/selected/{}.wav.scp'.format(loc,language)
ori_wav = '{}/{}.wav.scp'.format(loc,language)
op_wav = open(selected_wav,'w')
with open(ori_wav,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    utt = (line.split())[0]
    if utt in choosed_utt:
      op_wav.write(line+'\n')

op_wav.close()

#re-select ctm
selected_wrd_ctm = '{}/selected/{}_wrd.ctm'.format(loc,language)
selected_phn_ctm = '{}/selected/{}_phn.ctm'.format(loc,language)
ori_wrd_ctm = 'src/{}/{}_wrd.ctm'.format(language,language)
ori_phn_ctm = 'src/{}/{}_phn.ctm'.format(language,language)
op_wrd_ctm = open(selected_wrd_ctm,'w') 
with open(ori_wrd_ctm,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    utt = (line.split())[0]
    if utt in choosed_utt:
      op_wrd_ctm.write(line+'\n')
op_wrd_ctm.close()

op_phn_ctm = open(selected_phn_ctm,'w') 
with open(ori_phn_ctm,'r') as f:
  for line in f.readlines():
    line = line.rstrip()
    utt = (line.split())[0]
    if utt in choosed_utt:
      op_phn_ctm.write(line+'\n')
op_phn_ctm.close()


